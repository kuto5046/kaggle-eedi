import gc
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import polars as pl
from datasets import Dataset
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import wandb

LOGGER = logging.getLogger(__name__)


# ref: https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1#MAP@3-Metric
def mapk(preds: NDArray[np.object_], labels: NDArray[np.int_], k: int = 25) -> float:
    map_sum = 0
    for _x, y in zip(preds, labels):
        x = [int(i) for i in _x.split(" ")]
        z = [1 / i if y == j else 0 for i, j in zip(range(1, k + 1), x)]
        map_sum += np.sum(z)
    return map_sum / len(preds)


class TrainPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする

        # hydraのrun_dirに同じpathが設定されているので自動でディレクトリが作成される
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name
        self.common_cols = ["QuestionId", "ConstructName", "SubjectName", "QuestionText", "CorrectAnswer"]

        self.cfg = cfg
        self.oofs: list[pl.DataFrame] = []
        self.scores: list[float] = []
        self.debug_config()
        assert cfg.phase == "train", "TrainPipeline only supports train phase"

    def debug_config(self) -> None:
        if self.cfg.debug:
            self.cfg.trainer.epoch = 1
            self.cfg.trainer.save_steps = 0.5
            self.cfg.trainer.logging_steps = 0.5
            self.cfg.trainer.eval_steps = 0.5

    def setup_dataset(self, fold: int) -> None:
        df = pl.read_csv(self.cfg.path.feature_dir / self.cfg.feature_version / "train.csv")
        self.misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")

        if self.cfg.debug:
            df = df.sample(fraction=0.01)

        # dataset作成
        self.train = df.filter(pl.col("fold") != fold)
        self.valid = df.filter(pl.col("fold") == fold)
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

    def setup_logger(self, fold: int) -> None:
        wandb.init(  # type: ignore
            project="kaggle-eedi",
            entity="kuto5046",
            name=f"{self.cfg.exp_name}_{self.cfg.run_name}_fold{fold}",
            group=self.cfg.exp_name,
            tags=self.cfg.tags,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def training(self, fold: int) -> None:
        output_dir = self.output_dir / f"fold{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(self.cfg.model.name)

        loss = MultipleNegativesRankingLoss(self.model)
        params = self.cfg.trainer
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_dir,
            # Optional training parameters:
            num_train_epochs=params.epoch,
            per_device_train_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
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

        self.trainer = SentenceTransformerTrainer(
            model=self.model, args=args, train_dataset=self.train_dataset, eval_dataset=self.valid_dataset, loss=loss
        )

        self.trainer.train()
        self.model.save_pretrained(path=str(output_dir))

    def evaluate(self, fold: int) -> None:
        oof = self.valid.select(["QuestionId_Answer", "AllText", "MisconceptionId", "fold"]).unique()
        all_text_vec = self.model.encode(oof["AllText"].to_list(), normalize_embeddings=True)
        misconception_mapping_vec = self.model.encode(
            self.misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
        )
        similarity = cosine_similarity(all_text_vec, misconception_mapping_vec)
        sorted_similarity = np.argsort(-similarity, axis=1)
        oof = (
            oof.drop("AllText")
            .with_columns(pl.Series(sorted_similarity[:, : self.cfg.retrieve_num].tolist()).alias("pred"))
            .with_columns(pl.col("pred").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String))
        )
        oof.write_csv(self.output_dir / f"oof_{fold}.csv")
        score = mapk(preds=oof["pred"].to_numpy(), labels=oof["MisconceptionId"].to_numpy())
        LOGGER.info(f"CV: {score}")
        wandb.log({"CV": score})  # type: ignore
        self.oofs.append(oof)
        self.scores.append(score)

    def reset(self) -> None:
        if hasattr(self, "model"):
            del self.model

        if hasattr(self, "train_dataset"):
            del self.train_dataset

        if hasattr(self, "valid_dataset"):
            del self.valid_dataset

        if hasattr(self, "trainer"):
            del self.trainer

        gc.collect()
        torch.cuda.empty_cache()

    def run(self) -> None:
        for fold in self.cfg.use_folds:
            self.reset()
            self.setup_logger(fold)
            self.setup_dataset(fold)
            self.training(fold)
            self.evaluate(fold)
            # 最後のfold以外であればwandbをfinishする
            if fold != self.cfg.use_folds[-1]:
                wandb.finish()  # type: ignore

        oof = pl.concat(self.oofs).sort("QuestionId_Answer")
        oof.write_csv(self.output_dir / "oof.csv")
        score = mapk(preds=oof["pred"].to_numpy(), labels=oof["MisconceptionId"].to_numpy())
        LOGGER.info(f"ALL CV: {score}")
        wandb.log({"ALL CV": score})  # type: ignore
        wandb.finish()  # type: ignore


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
