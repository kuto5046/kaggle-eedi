import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


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


def calc_recall(df: pl.DataFrame) -> float:
    return (
        df.filter(pl.col("MisconceptionId") == pl.col("PredictMisconceptionId"))["QuestionId_Answer"].n_unique()
        / df["QuestionId_Answer"].n_unique()
    )


# ref: https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1#MAP@3-Metric
def mapk(preds: NDArray[np.object_], labels: NDArray[np.int_], k: int = 25) -> float:
    map_sum = 0
    for _x, y in zip(preds, labels):
        x = [int(i) for i in _x.split(" ")]
        z = [1 / i if y == j else 0 for i, j in zip(range(1, k + 1), x)]
        map_sum += np.sum(z)
    return map_sum / len(preds)


def calc_mapk(df: pl.DataFrame) -> float:
    agg_df = (
        df.group_by(["QuestionId_Answer"], maintain_order=True)
        .agg(pl.col("PredictMisconceptionId").alias("Predict"))
        .with_columns(pl.col("Predict").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String))
        .join(
            df[["QuestionId_Answer", "MisconceptionId"]].unique(),
            on=["QuestionId_Answer"],
        )
        .sort(by=["QuestionId_Answer"])
    )
    return mapk(agg_df["Predict"].to_numpy(), agg_df["MisconceptionId"])


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


def create_retrieved(df: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    df = (
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
        .with_row_index(name="idx")
    )
    return df


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

    def add_fold(self, df: pl.DataFrame) -> pl.DataFrame:
        return get_groupkfold(df, group_col="MisconceptionId", n_splits=self.cfg.n_splits)

    def generate_candidates(self, df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
        # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
        model = SentenceTransformer(self.cfg.retrieval_model.name)
        sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model)
        df = df.with_columns(
            pl.Series(sorted_similarity[:, : self.cfg.max_candidates].tolist()).alias("PredictMisconceptionId")
        )
        df = create_retrieved(df, misconception_mapping, self.cfg)
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df)
        if self.cfg.phase == "train":
            df = self.add_fold(df)
            df = self.generate_candidates(df, misconception)
            LOGGER.info(f"recall: {calc_recall(df):.5f}")
            LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
        df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
