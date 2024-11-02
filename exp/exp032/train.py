import os
import shutil
import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from datasets import Dataset
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from transformers import set_seed
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import wandb

from .data_processor import sentence_emb_similarity

LOGGER = logging.getLogger(__name__)

# seed固定用
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ref: https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1#MAP@3-Metric
def mapk(preds: NDArray[np.object_], labels: NDArray[np.int_], k: int = 25) -> float:
    map_sum = 0
    for _x, y in zip(preds, labels):
        x = [int(i) for i in _x.split(" ")]
        z = [1 / i if y == j else 0 for i, j in zip(range(1, k + 1), x)]
        map_sum += np.sum(z)
    return map_sum / len(preds)


# https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/539649
class CustomSentenceTransformer(SentenceTransformer):
    def _create_model_card(self, *args, **kwargs) -> None:  # type: ignore
        # Avoid to generate model card
        pass


class TrainPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        set_seed(cfg.seed, deterministic=True)
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.debug_config()

        # hydraのrun_dirに同じpathが設定されているので自動でディレクトリが作成される
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name

        assert cfg.phase == "train", "TrainPipeline only supports train phase"

    def debug_config(self) -> None:
        if self.cfg.debug:
            self.cfg.trainer.epoch = 1
            self.cfg.trainer.save_steps = 0.5
            self.cfg.trainer.logging_steps = 0.5
            self.cfg.trainer.eval_steps = 0.5
            self.cfg.exp_name = "dummy"
            self.cfg.run_name = "debug"

    def setup_dataset(self) -> None:
        df = pl.read_csv(self.cfg.path.feature_dir / self.cfg.feature_version / "train.csv")
        self.misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")

        if self.cfg.debug:
            df = df.sample(fraction=0.05, seed=self.cfg.seed)

        self.train = df.filter(pl.col("fold") != self.cfg.use_fold)
        self.valid = df.filter(pl.col("fold") == self.cfg.use_fold)
        # To create an anchor, positive, and negative structure,
        # delete rows where the positive and negative are identical.
        self.train_dataset = (
            Dataset.from_polars(self.train)
            .filter(
                lambda example: example["MisconceptionId"] != example["PredictMisconceptionId"],
            )
            .select_columns(
                # MisconceptionNameが正例、PredictMisconceptionNameが負例。ペアを1つの入力としている
                ["AllText", "MisconceptionName", "PredictMisconceptionName"]
            )
        )
        if self.cfg.shuffle_dataset:
            self.train_dataset = self.train_dataset.shuffle(seed=self.cfg.seed)

        self.valid_dataset = (
            Dataset.from_polars(self.valid)
            .filter(
                lambda example: example["MisconceptionId"] != example["PredictMisconceptionId"],
            )
            .select_columns(
                # MisconceptionNameが正例、PredictMisconceptionNameが負例。ペアを1つの入力としている
                ["AllText", "MisconceptionName", "PredictMisconceptionName"]
            )
        )

    def setup_logger(self) -> None:
        wandb.init(  # type: ignore
            project="kaggle-eedi",
            entity="kuto5046",
            name=f"{self.cfg.exp_name}_{self.cfg.run_name}",
            group=self.cfg.exp_name,
            tags=self.cfg.tags,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def training(self) -> None:
        self.model = CustomSentenceTransformer(self.cfg.retrieval_model.name, trust_remote_code=True)
        loss = MultipleNegativesRankingLoss(self.model)
        params = self.cfg.trainer
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=self.output_dir,
            # Optional training parameters:
            num_train_epochs=params.epoch,
            per_device_train_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            gradient_checkpointing=params.gradient_checkpointing,
            per_device_eval_batch_size=params.batch_size,
            eval_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            warmup_ratio=params.warmup_ratio,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            # Optional tracking/debugging parameters:
            lr_scheduler_type=params.lr_scheduler_type,
            save_strategy=params.save_strategy,
            save_steps=params.save_steps,
            save_total_limit=params.save_total_limit,
            logging_strategy=params.logging_strategy,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            metric_for_best_model=params.metric_for_best_model,
            report_to=params.report_to,
            run_name=self.cfg.exp_name + "_" + self.cfg.run_name,
            seed=self.cfg.seed,
            load_best_model_at_end=True,
            do_eval=True,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model, args=args, train_dataset=self.train_dataset, eval_dataset=self.valid_dataset, loss=loss
        )

        trainer.train()
        # checkpointを削除してbest modelを保存(save_strategyを有効にしていないとload_best_model_at_endが効かない)
        for ckpt_dir in (self.output_dir).glob(pattern="checkpoint-*"):
            shutil.rmtree(ckpt_dir)
        self.model.save_pretrained(path=str(self.output_dir))

    def evaluate(self) -> None:
        oof = self.valid.select(["QuestionId_Answer", "AllText", "MisconceptionId"]).unique()
        sorted_similarity = sentence_emb_similarity(oof, self.misconception_mapping, self.model)
        oof = (
            oof.drop("AllText")
            .with_columns(pl.Series(sorted_similarity[:, : self.cfg.retrieve_num].tolist()).alias("pred"))
            .with_columns(pl.col("pred").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String))
        )
        oof.write_csv(self.output_dir / "oof.csv")
        score = mapk(preds=oof["pred"].to_numpy(), labels=oof["MisconceptionId"].to_numpy())
        LOGGER.info(f"CV: {score}")
        wandb.log({"CV": score})  # type: ignore

    def run(self) -> None:
        self.setup_logger()
        self.setup_dataset()
        self.training()
        self.evaluate()
        wandb.finish()  # type: ignore


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
