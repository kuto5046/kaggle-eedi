import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

# trainデータに含まれるmisconceptionのユニーク数
ORIGINAL_TRAIN_UNIQUE_MISCONCEPTION_SIZE = 1604


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
    LOGGER.info(train.group_by("fold").len().sort("fold"))
    return train


def get_groupkfold(train: pl.DataFrame, target_col: str, n_splits: int) -> pl.DataFrame:
    kf = GroupKFold(n_splits=n_splits)
    cv = list(kf.split(X=train, groups=train[target_col].to_numpy()))
    return get_fold(train, cv)


def get_stratifiedgroupkfold(
    train: pl.DataFrame, target_col: str, group_col: str, n_splits: int, seed: int
) -> pl.DataFrame:
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train, y=train[target_col].to_numpy(), groups=train[group_col].to_numpy()))
    return get_fold(train, cv)


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


def adjust_unseen(
    seen: pl.DataFrame, unseen: pl.DataFrame, fold: int, unseen_valid_rate: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = seen.filter(pl.col("fold") != fold).drop("fold")
    valid = seen.filter(pl.col("fold") == fold).drop("fold")
    unseen_valid = unseen.sample(fraction=unseen_valid_rate, seed=seed)
    unseen_train = unseen.filter(~pl.col("QuestionId").is_in(unseen_valid["QuestionId"].to_list()))
    train = pl.concat([train, unseen_train])
    valid = pl.concat([valid, unseen_valid])
    unseen_rate = calcuate_unseen_rate(train, valid)
    valid_rate = len(valid) / (len(train) + len(valid))
    train_misconception_rate = len(train["MisconceptionId"].unique()) / ORIGINAL_TRAIN_UNIQUE_MISCONCEPTION_SIZE
    LOGGER.info(f"{fold=}: {unseen_rate=:.3%}, {valid_rate=:.3%} {train_misconception_rate=:.3%}")
    return train, valid


def calcuate_unseen_rate(train: pl.DataFrame, valid: pl.DataFrame) -> float:
    train_misconception_ids = train["MisconceptionId"].to_list()
    valid_misconception_ids = valid["MisconceptionId"].to_list()
    # unseen rateを計算
    unseen_misconception_ids = list(set(valid_misconception_ids) - set(train_misconception_ids))
    unseen_rate = len(unseen_misconception_ids) / len(valid_misconception_ids)
    return unseen_rate


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

    def preprocess(self, input_df: pl.DataFrame) -> pl.DataFrame:
        df = preprocess_table(input_df, self.common_cols)
        if self.cfg.phase == "test":
            return df
        else:
            # misconception情報(target)を取得
            pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
            df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
            df = df.filter(pl.col("MisconceptionId").is_not_null())
            return df

    def add_fold(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        # trainデータに1度しか出現していないmisconceptionIdを抽出し、unseenとして分ける
        candidate_misconception_ids = (
            df.group_by("MisconceptionId").len().filter(pl.col("len") == 1)["MisconceptionId"].to_list()
        )
        # 未知のmisconception用(unseen)として分けておく
        unseen = df.filter(pl.col("MisconceptionId").is_in(candidate_misconception_ids))

        # unseenを除くデータでfoldを作成
        # 実際にはseenにもunseenが含まれている。これにunseenを結合してunseen_rateを調整する
        seen = df.filter(~pl.col("QuestionId").is_in(unseen["QuestionId"].to_list()))
        seen = get_stratifiedgroupkfold(
            seen, target_col="MisconceptionId", group_col="QuestionId", n_splits=self.cfg.n_splits, seed=self.cfg.seed
        )
        return seen, unseen

    def generate_candidates(self, df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
        # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
        model = SentenceTransformer(self.cfg.model.name)
        sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model)
        df = df.with_columns(
            pl.Series(sorted_similarity[:, : self.cfg.retrieve_num].tolist()).alias("PredictMisconceptionId")
        )
        df = create_retrieved(df, misconception_mapping)
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df)
        if self.cfg.phase == "train":
            seen, unseen = self.add_fold(df)
            for fold in range(self.cfg.n_splits):
                train, valid = adjust_unseen(seen, unseen, fold, self.cfg.unseen_valid_rate, self.cfg.seed)
                # 必ずtrainとvalidでQuestionIdが被らないようにする
                assert len(set(train["QuestionId"].to_list()) & set(valid["QuestionId"].to_list())) == 0
                train = self.generate_candidates(train, misconception)
                valid = self.generate_candidates(valid, misconception)
                train.write_csv(self.output_dir / f"train_fold{fold}.csv")
                valid.write_csv(self.output_dir / f"valid_fold{fold}.csv")
                # hold-outで利用するのでfold=0のみ保存
                break
        else:
            df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
