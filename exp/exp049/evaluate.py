import logging
from pathlib import Path

import hydra
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig

from .inference import LLMPredictType, add_prompt, llm_inference
from .data_processor import (
    calc_mapk,
    calc_recall,
    get_groupkfold,
    preprocess_table,
    generate_candidates,
    get_stratifiedkfold,
    preprocess_misconception,
)

LOGGER = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)
        # if cfg.debug:
        #     cfg.run_name = "debug"
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
        tmp = df.with_row_index()
        df1 = tmp.sample(fraction=self.cfg.split_rate)
        df2 = tmp.filter(~pl.col("index").is_in(df1["index"]))
        df1 = get_groupkfold(df1, group_col="MisconceptionId", n_splits=self.cfg.n_splits)
        if len(df2) > 0:
            df2 = get_stratifiedkfold(df2, target_col="MisconceptionId", n_splits=self.cfg.n_splits, seed=self.cfg.seed)
            all_df = pl.concat([df1, df2])
        else:
            all_df = df1
        train = all_df.filter(pl.col("fold") != 0)
        valid = all_df.filter(pl.col("fold") == 0)

        train_misconception_ids = train["MisconceptionId"].to_list()
        valid_misconception_ids = valid["MisconceptionId"].to_list()
        unseen_misconceotion_ids = list(set(valid_misconception_ids) - set(train_misconception_ids))
        unseen_valid_size = valid.filter(pl.col("MisconceptionId").is_in(unseen_misconceotion_ids)).shape[0]
        unseen_rate = unseen_valid_size / valid.shape[0]
        LOGGER.info(f"unseen_rate: {unseen_rate=:.5f}")
        return all_df.drop("index").sort("QuestionId_Answer")

    def feature_engineering(self, df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = generate_candidates(df, misconception, self.cfg)
        LOGGER.info("first retrieval")
        LOGGER.info(f"fold={self.cfg.use_fold} validation size: {len(df)}")
        LOGGER.info(f"recall: {calc_recall(df):.5f}")
        LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
        if self.cfg.llm_model.use:
            df = add_prompt(df, misconception, self.cfg.llm_model.name, self.cfg.llm_model.predict_type)
            # LLMで予測
            df = llm_inference(df, self.cfg)
            if not self.cfg.llm_model.predict_type == LLMPredictType.RERANKING.value:
                df = generate_candidates(df, misconception, self.cfg)
                df.select(
                    [
                        "QuestionId_Answer",
                        "MisconceptionId",
                        "MisconceptionName",
                        "LLMPredictMisconceptionName",
                        "Prompt",
                    ]
                ).write_csv(self.output_dir / "eval.csv")

            LOGGER.info("second retrieval")
            LOGGER.info(f"fold={self.cfg.use_fold} validation size: {len(df)}")
            LOGGER.info(f"recall: {calc_recall(df):.5f}")
            LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df, misconception)
        df = self.add_fold(df)
        use_df = df.filter(pl.col("fold") == self.cfg.use_fold).sample(n=100, shuffle=True, seed=self.cfg.seed)
        use_df = self.feature_engineering(use_df, misconception)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
