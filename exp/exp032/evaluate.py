import logging
from pathlib import Path

import hydra
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig

from .inference import add_prompt, llm_inference
from .data_processor import (
    calc_mapk,
    calc_recall,
    get_groupkfold,
    preprocess_table,
    generate_candidates,
    preprocess_misconception,
)

LOGGER = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)
        if cfg.debug:
            cfg.run_name = "debug"
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name
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
        # misconception情報(target)を取得
        pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
        df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
        df = df.filter(pl.col("MisconceptionId").is_not_null())
        df = df.join(misconception, on="MisconceptionId", how="left")
        return df

    def add_fold(self, df: pl.DataFrame) -> pl.DataFrame:
        return get_groupkfold(df, group_col="QuestionId", n_splits=self.cfg.n_splits)

    def feature_engineering(self, df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = generate_candidates(
            df,
            misconception,
            self.cfg.retrieval_model.names,
            num_candidates=self.cfg.max_candidates,
            weights=self.cfg.retrieval_model.weights,
            local_files_only=True,
        )
        df = add_prompt(df, misconception, self.cfg.llm_model.name)
        # LLMで予測
        df = llm_inference(df, self.cfg)
        df.select(
            ["QuestionId_Answer", "MisconceptionId", "MisconceptionName", "LLMPredictMisconceptionName"]
        ).write_csv(self.output_dir / "eval.csv")
        df = generate_candidates(
            df,
            misconception,
            self.cfg.retrieval_model.names,
            num_candidates=self.cfg.retrieve_num,
            weights=self.cfg.retrieval_model.weights,
            local_files_only=True,
        )
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df, misconception)
        df = self.add_fold(df)
        use_df = df.filter(pl.col("fold") == self.cfg.use_fold).sample(n=100, shuffle=True, seed=self.cfg.seed)
        use_df = self.feature_engineering(use_df, misconception)
        LOGGER.info(f"fold={self.cfg.use_fold} validation size: {len(use_df)}")
        LOGGER.info(f"recall: {calc_recall(use_df):.5f}")
        LOGGER.info(f"mapk: {calc_mapk(use_df):.5f}")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
