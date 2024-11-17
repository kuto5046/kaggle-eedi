import gc
import logging
from pathlib import Path
from collections import defaultdict

import vllm
import hydra
import torch
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig

from .data_processor import add_prompt, generate_candidates

LOGGER = logging.getLogger(__name__)


def llm_inference(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    if cfg.llm_model.name in ["Qwen/Qwen2.5-Math-7B-Instruct"]:
        llm = vllm.LLM(**cfg.vllm.model)
    else:
        llm = vllm.LLM(**cfg.vllm.model)
    # tokenizer = llm.get_tokenizer()
    sampling_params = vllm.SamplingParams(**cfg.vllm.sampling)
    full_responses = llm.generate(
        prompts=df["Prompt"].to_numpy(),
        sampling_params=sampling_params,
        # lora_request=LoRARequest("adapter", 1, self.output_dir),
        use_tqdm=True,
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # question,idxをkeyとしてmisconception_idを取得する
    candidates = defaultdict(list)
    for row in df.iter_rows(named=True):
        candidates[row["QuestionId"]] = row["PredictMisconceptionId"]

    preds = []
    for x in full_responses:
        pred = ""
        for output in x.outputs:
            pred += output.text.replace("<|im_start|>", "").replace(":", "").strip() + " "
        preds.append(pred)
    df = df.with_columns(pl.Series(preds).alias("LLMPredictMisconceptionName")).with_columns(
        pl.concat_str(
            [
                pl.col("AllText"),
                pl.lit("\n## LLMPredictMisconception"),
                pl.col("LLMPredictMisconceptionName"),
            ],
            separator="",
        ).alias("AllText")
    )
    return df


class InferencePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        # assert cfg.phase == "test", "InferencePipeline only supports test phase"

    def setup_dataset(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.read_csv(self.cfg.path.feature_dir / self.cfg.feature_version / "test.csv")
        misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")

        group_cols = df.drop(["PredictMisconceptionId", "PredictMisconceptionName"]).columns
        df = df.group_by(group_cols, maintain_order=True).agg(
            pl.col("PredictMisconceptionId").alias("PredictMisconceptionId")
        )
        return df, misconception_mapping

    def make_submission(self, df: pl.DataFrame) -> None:
        submission = (
            df.with_columns(
                pl.col("PredictMisconceptionId")
                .map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String)
                .alias("MisconceptionId")
            )
            .filter(pl.col("CorrectAnswerAlphabet") != pl.col("InCorrectAnswerAlphabet"))
            .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
            .sort("QuestionId_Answer")
        )
        submission.write_csv(self.cfg.path.sub_dir / "submission.csv")
        # アンサンブル用
        output_dir = self.cfg.path.output_dir / self.cfg.exp_name / self.cfg.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        submission.write_csv(output_dir / "submission.csv")

    def run(self) -> None:
        # embモデルでfirst retrieval
        df, misconception_mapping = self.setup_dataset()
        if self.cfg.llm_model.use:
            # llm inference
            df = add_prompt(df, misconception_mapping, self.cfg.llm_model.name)
            df = llm_inference(df, self.cfg)
            # second retreval
            df = generate_candidates(
                df,
                misconception_mapping,
                self.cfg,
            )
        self.make_submission(df)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
