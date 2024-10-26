import re
import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

# trainデータに含まれるmisconceptionのユニーク数
ORIGINAL_TRAIN_UNIQUE_MISCONCEPTION_SIZE = 1604


def preprocess_text(x: str) -> str:
    x = re.sub(r"http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


def add_prompt(df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
    prompt = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher.
Your task is to reason and identify the misconception behind the Incorrect Answer with the Question in English.
Answer concisely what misconception it is to lead to getting the incorrect answer.
No need to give the reasoning process and do not use "The misconception is" to start your answers.
There are some relative and possible misconceptions below to help you make the decision:

{Retrieval}
"""
    id2name_mapping = {row["MisconceptionId"]: row["MisconceptionName"] for row in misconception.iter_rows(named=True)}
    texts = [
        preprocess_text(
            prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                CorrectAnswer=row["CorrectAnswerText"],
                IncorrectAnswer=row["InCorrectAnswerText"],
                MisconceptionName=row["MisconceptionName"],
                Retrieval=get_retrieval_text(row["PredictMisconceptionId"], id2name_mapping),
            )
        )
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    return df


def get_retrieval_text(misconception_ids: list[int], id2name_mapping: dict[int, str]) -> str:
    retrieval = ""
    for i, id in enumerate(misconception_ids):
        name = id2name_mapping[id]
        retrieval += f"- {name} \n"
    return retrieval


def preprocess_table(df: pl.DataFrame, common_cols: list[str]) -> pl.DataFrame:
    long_df = (
        df.select(pl.col(common_cols + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_cols,
            variable_name="AnswerType",
            value_name="AnswerText",
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("ConstructName"),
                    pl.col("SubjectName"),
                    pl.col("QuestionText"),
                    pl.col("AnswerText"),
                ],
                separator=" ",
            ).alias("AllText"),
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
    )
    # 問題-正解-不正解のペアを作る
    correct_df = (
        long_df.filter(pl.col("CorrectAnswer") == pl.col("AnswerAlphabet"))
        .select(["QuestionId", "AnswerAlphabet", "AnswerText"])
        .rename({"AnswerAlphabet": "CorrectAnswerAlphabet", "AnswerText": "CorrectAnswerText"})
    )
    long_df = (
        long_df.join(correct_df, on=["QuestionId"], how="left")
        .rename({"AnswerAlphabet": "InCorrectAnswerAlphabet", "AnswerText": "InCorrectAnswerText"})
        .filter(pl.col("InCorrectAnswerAlphabet") != pl.col("CorrectAnswerAlphabet"))
    )
    return long_df


def preprocess_misconception(df: pl.DataFrame, common_cols: list[str]) -> pl.DataFrame:
    misconception = (
        df.select(pl.col(common_cols + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_cols,
            variable_name="MisconceptionType",
            value_name="MisconceptionId",
        )
        .with_columns(
            pl.col("MisconceptionType").str.extract(r"Misconception([A-D])Id$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
        .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )
    return misconception


def get_fold(_train: pl.DataFrame, cv: list[tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    """
    trainにfoldのcolumnを付与する
    """
    train = _train.clone()
    train = train.with_columns(pl.lit(-1).alias("fold"))
    for fold, (train_idx, valid_idx) in enumerate(cv):
        train = train.with_columns(
            pl.when(pl.arange(0, len(train)).is_in(valid_idx)).then(fold).otherwise(pl.col("fold")).alias("fold")
        )
    LOGGER.info(train.group_by("fold").len().sort("fold"))
    return train


def get_groupkfold(train: pl.DataFrame, group_col: str, n_splits: int) -> pl.DataFrame:
    kf = GroupKFold(n_splits=n_splits)
    cv = list(kf.split(X=train, groups=train[group_col].to_numpy()))
    return get_fold(train, cv)


def generate_candidates(df: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
    df_list = []
    for fold in range(cfg.n_splits):
        # リークしないようにoofのモデルで候補を生成
        model_dir = str(cfg.path.model_dir / cfg.retrieval_model.pretrained_exp_name / f"fold{fold}")
        model = SentenceTransformer(model_dir)  # foldを見ていないモデル
        fold_df = df.filter(pl.col("fold") == fold)  # foldのデータ
        sorted_similarity = sentence_emb_similarity(fold_df, misconception_mapping, model)
        fold_df = fold_df.with_columns(
            pl.Series(sorted_similarity[:, : cfg.max_candidates].tolist()).alias("PredictMisconceptionId")
        )
        df_list.append(fold_df)
    output_df = pl.concat(df_list).filter(pl.col("MisconceptionId").is_not_null()).sort("QuestionId_Answer")
    return output_df


def sentence_emb_similarity(
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    text_vec = model.encode(df["AllText"].to_list(), normalize_embeddings=True)
    misconception_mapping_vec = model.encode(
        misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
    )
    similarity = cosine_similarity(text_vec, misconception_mapping_vec)
    sorted_similarity = np.argsort(-similarity, axis=1)
    return sorted_similarity


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        self.output_dir = cfg.path.feature_dir / cfg.feature_version
        self.output_dir.mkdir(exist_ok=True, parents=True)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.common_cols = ["QuestionId", "ConstructName", "SubjectName", "QuestionText", "CorrectAnswer"]

    def read_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.read_csv(self.cfg.path.input_dir / f"{self.cfg.phase}.csv")
        misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")
        return df, misconception_mapping

    def preprocess(self, input_df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = preprocess_table(input_df, self.common_cols)
        if self.cfg.phase == "test":
            return df
        else:
            # misconception情報(target)を取得
            pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
            df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
            df = df.filter(pl.col("MisconceptionId").is_not_null())
            df = df.join(misconception, on="MisconceptionId", how="left")
            return df

    def add_fold(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        return get_groupkfold(df, group_col="QuestionId", n_splits=self.cfg.n_splits)

    def feature_engineering(self, df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = generate_candidates(df, misconception, self.cfg)
        df = add_prompt(df, misconception)
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df, misconception)
        if self.cfg.phase == "train":
            df = self.add_fold(df)
        df = self.feature_engineering(df, misconception)
        df = df.select(["fold", "QuestionId_Answer", "MisconceptionId", "MisconceptionName", "Prompt"])
        df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
