from pathlib import Path

import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity


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
    print(train.group_by("fold").len().sort("fold"))
    return train


def get_groupkfold(train: pl.DataFrame, target_col: str, n_splits: int) -> pl.DataFrame:
    kf = GroupKFold(n_splits=n_splits)
    cv = list(kf.split(X=train, groups=train[target_col].to_numpy()))
    return get_fold(train, cv)


def sentence_emb_similarity(
    df: pl.DataFrame, misconception_mapping: pl.DataFrame, model: SentenceTransformer, cfg: DictConfig
) -> np.ndarray:
    text_vec = model.encode(df["AllText"].to_list(), normalize_embeddings=True)
    misconception_mapping_vec = model.encode(
        misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
    )
    similarity = cosine_similarity(text_vec, misconception_mapping_vec)
    sorted_similarity = np.argsort(-similarity, axis=1)
    return sorted_similarity


def create_retrieved(df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
    retrieved_df = (
        df.filter(
            pl.col("MisconceptionId").is_not_null()  # TODO: Consider ways to utilize data where MisconceptionId is NaN.
        )
        .explode("PredictMisconceptionId")
        .join(
            misconception_mapping,
            on="MisconceptionId",
        )
        .join(
            misconception_mapping.rename(lambda x: "Predict" + x),
            on="PredictMisconceptionId",
        )
    )
    return retrieved_df


def upsampling_unseen(
    valid: pl.DataFrame, unseen_misconception_ids: list[int], unseen_rate: float, seed: int
) -> pl.DataFrame:
    seen_valid = valid.filter(~pl.col("MisconceptionId").is_in(unseen_misconception_ids))
    unseen_valid = valid.filter(pl.col("MisconceptionId").is_in(unseen_misconception_ids))
    current_unseen_rate = unseen_valid.shape[0] / valid.shape[0]
    # unseen_rateがtarget_unseen_rateを超えるまでunseen_validから復元抽出
    assert current_unseen_rate <= unseen_rate

    # unseenをアップサンプリングしてtarget_unseen_rateにする
    need_size = int((unseen_rate * valid.shape[0] - unseen_valid.shape[0]) / (1 - unseen_rate))
    rng = np.random.default_rng(seed)
    sampling_index = rng.integers(0, len(unseen_valid), need_size)
    aug_unseen_valid = []
    for idx in sampling_index:
        aug_unseen_valid.append(unseen_valid.row(idx))
    aug_unseen_valid = pl.DataFrame(aug_unseen_valid, schema=unseen_valid.schema)
    unseen_valid = pl.concat([unseen_valid, aug_unseen_valid])

    valid = pl.concat([seen_valid, unseen_valid])
    return valid


def adjust_unseen_rate(df: pl.DataFrame, fold: int, unseen_rate: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("fold") != fold)
    valid = df.filter(pl.col("fold") == fold)

    train_misconception_ids = train["MisconceptionId"].to_list()
    valid_misconception_ids = valid["MisconceptionId"].to_list()
    # validにしか存在しない未知のmisconception_idを取得
    unseen_misconceotion_ids = list(set(valid_misconception_ids) - set(train_misconception_ids))

    valid = upsampling_unseen(valid, unseen_misconceotion_ids, unseen_rate, seed)
    unseen_valid_size = valid.filter(pl.col("MisconceptionId").is_in(unseen_misconceotion_ids)).shape[0]
    unseen_rate = unseen_valid_size / valid.shape[0]
    print(f"fold{fold}: unseen_misconception_rate={unseen_rate:.4f}")
    return train, valid


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

    def preprocess(self, input_df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
        df = preprocess_table(input_df, self.common_cols)
        if self.cfg.phase == "test":
            return df

        # misconception情報(target)を取得
        pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
        df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")

        # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
        model = SentenceTransformer(self.cfg.model.name)
        sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model, self.cfg)
        df = df.with_columns(
            pl.Series(sorted_similarity[:, : self.cfg.retrieve_num].tolist()).alias("PredictMisconceptionId")
        )
        df = create_retrieved(df, misconception_mapping)
        return df

    def add_fold(self, df: pl.DataFrame) -> None:
        if self.cfg.phase == "train":
            df = get_groupkfold(df, "QuestionId", self.cfg.n_splits)
            for fold in range(self.cfg.n_splits):
                train, valid = adjust_unseen_rate(df, fold, self.cfg.valid_unseen_misconception_rate, self.cfg.seed)
                # question_idに重複はない
                assert set(train["QuestionId"].to_list()) & set(valid["QuestionId"].to_list()) == set()
                train.write_csv(self.output_dir / f"train_fold{fold}.csv")
                valid.write_csv(self.output_dir / f"valid_fold{fold}.csv")
        else:
            df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df, misconception)
        self.add_fold(df)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
