import gc
import os
import logging
from typing import Union
from pathlib import Path

import hydra
import numpy as np
import torch
import polars as pl
from peft import PeftModel
from torch import nn
from datasets import Dataset as HFDataset
from lightning import seed_everything
from omegaconf import DictConfig
from transformers import (
    Trainer,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from sentence_transformers import SentenceTransformer, util
from transformers.file_utils import ModelOutput
from sklearn.metrics.pairwise import cosine_similarity

import wandb

from .data_processor import apk, encode, calc_mapk, calc_recall, setup_qlora_model, sentence_emb_similarity_by_peft

LOGGER = logging.getLogger(__name__)

TOKENIZER = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
MODEL = Union[PreTrainedModel, SentenceTransformer, nn.Module]

# seed固定用
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CustomCallback(TrainerCallback):
    def __init__(
        self, trainer: Trainer, valid: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig
    ) -> None:
        super().__init__()
        self._trainer = trainer
        self.valid = valid.select(["QuestionId_Answer", "AllText", "MisconceptionId"]).unique()
        self.misconception_mapping = misconception_mapping
        self.cfg = cfg

    def on_evaluate(self, args, state, control, **kwargs) -> None:  # type: ignore
        model = self._trainer.model.model
        tokenizer = self._trainer.tokenizer
        query_embs = encode(
            model,
            tokenizer,
            self.valid["AllText"].to_list(),
            model_name=self.cfg.retrieval_model.name,
            batch_size=self.cfg.trainer.batch_size,
        )
        passage_embs = encode(
            model,
            tokenizer,
            self.misconception_mapping["MisconceptionName"].to_list(),
            model_name=self.cfg.retrieval_model.name,
            batch_size=self.cfg.trainer.batch_size,
        )
        similarity = cosine_similarity(query_embs, passage_embs)
        sorted_similarity = np.argsort(-similarity, axis=1)
        submission = (
            self.valid.with_columns(
                pl.Series(sorted_similarity[:, :25].tolist()).alias("MisconceptionIdPred"),
            )
            .with_columns(
                pl.col("MisconceptionIdPred")
                .map_elements(
                    lambda x: " ".join(map(str, x)),
                    return_dtype=pl.String,
                )
                .str.split(" "),
            )
            .with_columns(
                pl.col("MisconceptionId").cast(pl.Int64).cast(pl.String).str.split(" ").alias("MisconceptionIdGT"),
            )
            .select(
                pl.col(["QuestionId_Answer", "MisconceptionIdGT", "MisconceptionIdPred"]),
            )
            .sort("QuestionId_Answer")
        )

        submission = submission.to_pandas()
        submission["ap@25"] = submission.apply(
            lambda x: apk(x["MisconceptionIdGT"], x["MisconceptionIdPred"], k=25),
            axis=1,
        )
        LOGGER.info(f"valid map@25: {submission['ap@25'].mean()}")
        wandb.log({"eval/map@25": submission["ap@25"].mean()})  # type: ignore


# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L12-L124
class CustomMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0) -> None:
        super().__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor_embs: torch.tensor,
        pos_embs: torch.tensor,
        neg_embs: torch.tensor,
        pos_ids: torch.tensor,
        neg_ids: torch.tensor,
    ) -> torch.tensor:
        cand_embs = torch.cat([pos_embs, neg_embs])  # (batch*2, emb_dim)
        ids = torch.cat([pos_ids, neg_ids])  # (batch*2)
        # scores[i][j]: i番目のanchorとj番目のcandidateのcos類似度
        scores = util.cos_sim(anchor_embs, cand_embs) * self.scale  # (batch, batch*2)

        # 対角成分と一致するかどうかを判定するためのmask
        mask = torch.eq(pos_ids.unsqueeze(1), ids.unsqueeze(0))  # (batch, batch*2)
        # 対角成分はFalseにする
        mask.fill_diagonal_(False)
        # mask部分を-infにして負例として扱わないようにする
        scores = scores.masked_fill(mask, float("-inf"))
        range_labels = torch.arange(0, scores.size(0), device=scores.device)
        return self.cross_entropy_loss(scores, range_labels)


class TripletCollator:
    def __init__(self, tokenizer: TOKENIZER, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.tensor]:
        queries = [f["AllText"] for f in features]
        positives = [f["MisconceptionName"] for f in features]
        negatives = [f["PredictMisconceptionName"] for f in features]
        positive_ids = [f["MisconceptionId"] for f in features]
        negative_ids = [f["PredictMisconceptionId"] for f in features]
        # Tokenize each of the triplet components separately
        queries_encoded = self.tokenizer(
            queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        positives_encoded = self.tokenizer(
            positives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        negatives_encoded = self.tokenizer(
            negatives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        device = queries_encoded["input_ids"].device
        return {
            "anchor": queries_encoded,
            "positive": positives_encoded,
            "negative": negatives_encoded,
            "positive_id": torch.tensor(positive_ids, device=device),
            "negative_id": torch.tensor(negative_ids, device=device),
        }


class TripletSimCSEModel(nn.Module):
    def __init__(self, model: PeftModel, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.criterion = CustomMultipleNegativesRankingLoss()
        self.retrieval_model_name = cfg.retrieval_model.name

    def sentence_embedding(self, hidden_state: torch.tensor, mask: torch.tensor) -> torch.tensor:
        return hidden_state[torch.arange(hidden_state.size(0)), mask.sum(1) - 1]

    def encode(self, features: dict[str, torch.tensor]) -> torch.tensor:
        if self.retrieval_model_name == "nvidia/NV-Embed-v2":
            outputs = self.model(**features)
            return self.sentence_embedding(outputs["sentence_embeddings"], features["attention_mask"])
        else:
            outputs = self.model(**features)
            return self.sentence_embedding(outputs.last_hidden_state, features["attention_mask"])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict) -> None:
        self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(
        self,
        anchor: dict[str, torch.tensor],
        positive: dict[str, torch.tensor],
        negative: dict[str, torch.tensor],
        positive_id: list[int],
        negative_id: list[int],
    ) -> ModelOutput:
        anchor_emb = self.encode(anchor)  # Anchor embeddings
        pos_emb = self.encode(positive)  # Positive embeddings
        neg_emb = self.encode(negative)  # Negative embeddings

        # Compute triplet loss
        loss = self.criterion(anchor_emb, pos_emb, neg_emb, positive_id, negative_id)
        return ModelOutput(loss=loss, anchor=anchor_emb, positive=pos_emb, negative=neg_emb)


class TripletTrainer(Trainer):
    def compute_loss(
        self, model: MODEL, inputs: dict[str, torch.tensor], return_outputs: bool = False
    ) -> Union[torch.tensor, ModelOutput]:
        # Only pass anchor, positive, and negative to the model
        outputs = model(
            inputs["anchor"], inputs["positive"], inputs["negative"], inputs["positive_id"], inputs["negative_id"]
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


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

        # group化することでQuestionと正例のペアがバッチ内で重複しないようにする
        # df = (
        #     df.filter(pl.col("MisconceptionId") != pl.col("PredictMisconceptionId"))
        #     .group_by(
        #         ["QuestionId_Answer", "AllText", "MisconceptionName", "MisconceptionId", "fold"], maintain_order=True
        #     )
        #     .agg(pl.col("PredictMisconceptionName").alias("PredictMisconceptionName"))
        # )
        if self.cfg.debug:
            df = df.sample(fraction=0.01, seed=self.cfg.seed)

        self.train = df.filter(pl.col("fold") != self.cfg.use_fold)
        self.valid = df.filter(pl.col("fold") == self.cfg.use_fold)

        self.train_dataset = HFDataset.from_polars(self.train).select_columns(
            # MisconceptionNameが正例、PredictMisconceptionNameが負例。ペアを1つの入力としている
            ["AllText", "MisconceptionName", "PredictMisconceptionName", "MisconceptionId", "PredictMisconceptionId"]
        )
        if self.cfg.shuffle_dataset:
            self.train_dataset = self.train_dataset.shuffle(seed=self.cfg.seed)

        self.valid_dataset = HFDataset.from_polars(self.valid).select_columns(
            ["AllText", "MisconceptionName", "PredictMisconceptionName", "MisconceptionId", "PredictMisconceptionId"]
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
        lora_model, tokenizer = setup_qlora_model(self.cfg, pretrained_lora_path=None)
        model = TripletSimCSEModel(lora_model, self.cfg)

        data_collator = TripletCollator(tokenizer)

        params = self.cfg.trainer
        args = TrainingArguments(
            # Required parameter:
            output_dir=self.output_dir,
            # Optional training parameters:
            num_train_epochs=params.epoch,
            per_device_train_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            gradient_checkpointing=params.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=params.batch_size,
            eval_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            warmup_ratio=params.warmup_ratio,
            fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=True,  # Set to True if you have a GPU that supports BF16
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
            load_best_model_at_end=False,
            do_eval=True,
            remove_unused_columns=False,
        )

        trainer = TripletTrainer(
            model=model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.add_callback(
            CustomCallback(trainer, self.valid, self.misconception_mapping, self.cfg)
        )  # <-- just add one line

        trainer.can_return_loss = True  # peft modelを利用するとeval_lossが出力されないバグがあるため一時的な対応
        trainer.train()
        # checkpointを削除してbest modelを保存(save_strategyを有効にしていないとload_best_model_at_endが効かない)
        # for ckpt_dir in (self.output_dir).glob(pattern="checkpoint-*"):
        #     shutil.rmtree(ckpt_dir)
        # LoRAモデルを保存
        model.model.save_pretrained(str(self.output_dir))

        del model, trainer, lora_model
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self) -> None:
        oof = self.valid.select(["QuestionId_Answer", "AllText", "MisconceptionId"]).unique()
        sorted_similarity = sentence_emb_similarity_by_peft(
            oof,
            self.misconception_mapping,
            self.cfg,
            pretrained_lora_path=self.output_dir,
        )
        oof = oof.with_columns(
            pl.Series(sorted_similarity[:, : self.cfg.retrieve_num].tolist()).alias("PredictMisconceptionId")
        )
        recall = calc_recall(oof)
        mapk = calc_mapk(oof)
        LOGGER.info(f"Recall: {recall:.5f}")
        LOGGER.info(f"CV: {mapk:.5f}")
        wandb.log({"Recall": recall})  # type: ignore
        wandb.log({"CV": mapk})  # type: ignore
        oof = oof.drop("AllText").with_columns(
            pl.col("PredictMisconceptionId").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String)
        )
        oof.write_csv(self.output_dir / "oof.csv")

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
