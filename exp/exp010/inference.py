import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from .data_processor import sentence_emb_similarity

LOGGER = logging.getLogger(__name__)


# https://www.kaggle.com/code/titericz/h-m-ensembling-how-to
def ensemble_predictions(preds: list[np.ndarray], weights: list[float] | None = None, top_k: int = 25) -> np.ndarray:
    if weights is None:
        weights = [1] * len(preds)

    sample_size = preds[0].shape[0]
    blend_results = []
    for i in range(sample_size):
        scores: dict[int, float] = {}
        for j in range(len(preds)):
            w = weights[j]
            # 順位に応じて重みをつける
            for k, pred_misconception_id in enumerate(preds[j][i]):
                if pred_misconception_id in scores:
                    scores[pred_misconception_id] += w / (k + 1)
                else:
                    scores[pred_misconception_id] = w / (k + 1)
        # Sort dictionary by item weights
        result = list(dict(sorted(scores.items(), key=lambda item: -item[1])).keys())[:top_k]
        blend_results.append(result)
    return np.array(blend_results)


class InferencePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.model_dir = self.cfg.path.model_dir / self.cfg.exp_name / self.cfg.run_name
        assert cfg.phase == "test", "InferencePipeline only supports test phase"

    def setup_dataset(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.read_csv(self.cfg.path.feature_dir / self.cfg.feature_version / "test.csv")
        misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")
        return df, misconception_mapping

    def setup_model(self) -> SentenceTransformer:
        return SentenceTransformer(str(self.model_dir))

    def inference(
        self, model: SentenceTransformer, df: pl.DataFrame, misconception_mapping: pl.DataFrame
    ) -> np.ndarray:
        sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model, self.cfg)
        return sorted_similarity

    def make_submission(self, df: pl.DataFrame, preds: list[np.ndarray]) -> None:
        # アンサンブル用に絞る(25より大きめにしとく)
        filter_preds = [pred[:, : self.cfg.retrieve_num + 10] for pred in preds]
        pred = ensemble_predictions(filter_preds)
        submission = (
            df.with_columns(pl.Series(pred[:, : self.cfg.retrieve_num].tolist()).alias("MisconceptionId"))
            .with_columns(
                pl.col("MisconceptionId").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String)
            )
            .filter(pl.col("CorrectAnswer") != pl.col("AnswerAlphabet"))
            .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
            .sort("QuestionId_Answer")
        )
        submission.write_csv(self.cfg.path.sub_dir / "submission.csv")
        # アンサンブル用
        output_dir = self.cfg.path.output_dir / self.cfg.exp_name / self.cfg.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        submission.write_csv(output_dir / "submission.csv")

    def run(self) -> None:
        df, misconception_mapping = self.setup_dataset()
        preds = []
        model = self.setup_model()
        pred = self.inference(model, df, misconception_mapping)
        preds.append(pred)
        self.make_submission(df, preds)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
