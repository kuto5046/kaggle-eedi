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
from tokenizers import AddedToken
from numpy.typing import NDArray
from transformers import (
    Trainer,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    set_seed,
)
from scipy.special import softmax
from sklearn.metrics import log_loss

import wandb

NUM_LABELS = 2

LOGGER = logging.getLogger(__name__)

# seed固定用
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ref: https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1#MAP@3-Metric
def mapk(preds: NDArray[np.object_], labels: NDArray[np.int_], k: int = 25) -> float:
    map_sum = 0
    for _x, y in zip(preds, labels):
        x = [int(i) for i in _x.split(" ")]
        z = [1 / i if y == j else 0 for i, j in zip(range(1, k + 1), x)]
        map_sum += np.sum(z)
    return map_sum / len(preds)


def tokenize(examples: dict[str, str], max_token_length: int, tokenizer: AutoTokenizer) -> dict[str, list]:
    separator = " [SEP] "

    joined_text = (
        examples["ConstructName"]
        + separator
        + examples["SubjectName"]
        + separator
        + examples["QuestionText"]
        + separator
        + examples["AnswerText"]
        + separator  # TODO: use other special token
        + examples["PredictMisconceptionName"]
    )

    return tokenizer(
        joined_text,
        max_length=max_token_length,
        truncation=True,
        padding="max_length",
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred
    preds_prob = softmax(predictions, axis=-1)
    return {"eval_loss": log_loss(labels, preds_prob)}


class TrainPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        # cfg.pathの中身をPathに変換する
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        set_seed(cfg.seed, deterministic=True)
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする

        # hydraのrun_dirに同じpathが設定されているので自動でディレクトリが作成される
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name

        self.cfg = cfg
        self.debug_config()
        assert cfg.phase == "train", "TrainPipeline only supports train phase"

    def debug_config(self) -> None:
        if self.cfg.debug:
            self.cfg.trainer.epoch = 1
            self.cfg.trainer.save_steps = 0.5
            self.cfg.trainer.logging_steps = 0.5
            self.cfg.trainer.eval_steps = 0.5

    def setup_dataset(self) -> None:
        self.train = pl.read_csv(
            self.cfg.path.feature_dir / self.cfg.feature_version / f"train_fold{self.cfg.use_fold}.csv"
        )
        self.valid = pl.read_csv(
            self.cfg.path.feature_dir / self.cfg.feature_version / f"valid_fold{self.cfg.use_fold}.csv"
        )
        self.misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")

        if self.cfg.debug:
            self.train = self.train.sample(fraction=0.05, seed=self.cfg.seed)
            self.valid = self.valid.sample(fraction=0.05, seed=self.cfg.seed)

        self.train = self.train.with_columns(
            (pl.col("MisconceptionId") == pl.col("PredictMisconceptionId")).cast(pl.Int16).alias("label")
        )
        self.valid = self.valid.with_columns(
            (pl.col("MisconceptionId") == pl.col("PredictMisconceptionId")).cast(pl.Int16).alias("label")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.reranker_model.name)
        self.tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        self.tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])
        self.train_dataset = Dataset.from_polars(self.train).map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": self.tokenizer, "max_token_length": 256},
            num_proc=4,
        )

        self.valid_dataset = Dataset.from_polars(self.valid).map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": self.tokenizer, "max_token_length": 256},
            num_proc=4,
        )

    def setup_logger(self) -> None:
        wandb.init(  # type: ignore
            project="kaggle-eedi",
            entity="kuto5046",
            name=f"{self.cfg.exp_name}_{self.cfg.run_name}_fold{self.cfg.use_fold}",
            group=self.cfg.exp_name,
            tags=self.cfg.tags,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def training(self) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.reranker_model.name, num_labels=NUM_LABELS)
        model.resize_token_embeddings(len(self.tokenizer))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=16)
        params = self.cfg.trainer
        args = TrainingArguments(
            # Required parameter:
            output_dir=self.output_dir,
            # Optional training parameters:
            num_train_epochs=params.epoch,
            per_device_train_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            per_device_eval_batch_size=int(params.batch_size * 2),
            eval_accumulation_steps=params.gradient_accumulation_steps // 2,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            warmup_ratio=params.warmup_ratio,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
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
            # label_names=['label'],  # 指定するとeval_lossがmetricに反映されずerrorとなる
        )

        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
        # checkpointを削除してbest modelを保存(save_strategyを有効にしていないとload_best_model_at_endが効かない)
        for ckpt_dir in (self.output_dir).glob(pattern="checkpoint-*"):
            shutil.rmtree(ckpt_dir)
        model.save_pretrained(str(self.output_dir))
        # self.trainer.save_model(str(self.output_dir))

    def evaluate(self) -> None:
        preds = softmax(self.trainer.predict(self.valid_dataset).predictions, axis=-1)

        def add_valid_pred(example: dict, idx: int, preds: np.ndarray) -> dict:
            example["pred"] = preds[idx]
            return example

        oof = (
            self.valid.with_columns(pl.Series(preds[:, 1]).alias("pred"))
            .sort(by=["QuestionId_Answer", "pred"], descending=[False, True])
            .group_by(["QuestionId_Answer"], maintain_order=True)
            .agg(pl.col("PredictMisconceptionId").alias("Predict"))
            .with_columns(pl.col("Predict").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String))
            .join(
                self.valid_dataset.to_polars()[["QuestionId_Answer", "MisconceptionId"]].unique(),
                on=["QuestionId_Answer"],
            )
            .sort(by=["QuestionId_Answer"])
        )

        oof.write_csv(self.output_dir / "oof.csv")
        score = mapk(preds=oof["Predict"].to_numpy(), labels=oof["MisconceptionId"].to_numpy())
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
