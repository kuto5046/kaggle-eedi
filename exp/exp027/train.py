import gc
import os
import shutil
import logging
from pathlib import Path

import vllm
import hydra
import numpy as np
import torch
import polars as pl
from trl import DataCollatorForCompletionOnlyLM
from peft import TaskType, LoraConfig, get_peft_model
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from transformers import (
    Trainer,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForCausalLM,
    set_seed,
)
from scipy.special import softmax
from sklearn.metrics import log_loss
from vllm.lora.request import LoRARequest

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
            self.cfg.run_name = "debug"

    def setup_dataset(self) -> None:
        df = pl.read_csv(self.cfg.path.feature_dir / self.cfg.feature_version / "train.csv")
        self.misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")

        self.train = df.filter(pl.col("fold") != self.cfg.use_fold)
        self.valid = df.filter(pl.col("fold") == self.cfg.use_fold)

        if self.cfg.debug:
            self.train = self.train.sample(fraction=0.05, seed=self.cfg.seed)
            self.valid = self.valid.sample(fraction=0.05, seed=self.cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm_model.name)

        self.train_dataset = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["Prompt"]},
                    {"role": "assistant", "content": f'{row["MisconceptionName"]}'},
                ],
                tokeadd_generation_prompt=True,
            )
            for row in self.train.iter_rows(named=True)
        ]

        self.valid_dataset = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["Prompt"]},
                    {"role": "assistant", "content": f'{row["MisconceptionName"]}'},
                ],
                tokeadd_generation_prompt=True,
            )
            for row in self.valid.iter_rows(named=True)
        ]

        self.eval_dataset = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["Prompt"]},
                ],
                tokeadd_generation_prompt=True,
                tokenize=False,  # textとして渡す
            )
            + "<|im_start|>assistant"
            for row in self.valid.iter_rows(named=True)
        ]

    def setup_logger(self) -> None:
        wandb.init(  # type: ignore
            project="kaggle-eedi",
            entity="kuto5046",
            name=f"{self.cfg.exp_name}_{self.cfg.run_name}",
            group=self.cfg.exp_name,
            # tags=self.cfg.tags,
            mode="disabled" if self.cfg.debug else "online",
            notes=self.cfg.notes,
        )

    def training(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.llm_model.name,
            torch_dtype=torch.float16,
            # quantization_config=quantization_config,
            use_cache=False,
            # attn_implementation="flash_attention_2",  # メモリ節約目的だが変わらず
            device_map="auto",
        )

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant", tokenizer=self.tokenizer
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules={
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            },
            **self.cfg.llm_model.lora,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        LOGGER.info(model.print_trainable_parameters())

        params = self.cfg.trainer
        args = TrainingArguments(
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
            fp16=params.fp16,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=params.bf16,  # Set to True if you have a GPU that supports BF16
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
            gradient_checkpointing_kwargs={"use_reentrant": False},
            group_by_length=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        _ = trainer.train()
        # checkpointを削除してbest modelを保存(save_strategyを有効にしていないとload_best_model_at_endが効かない)
        for ckpt_dir in (self.output_dir).glob(pattern="checkpoint-*"):
            shutil.rmtree(ckpt_dir)

        # LoRA adaptorのみ保存
        model.save_pretrained(str(self.output_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(self.output_dir))
        del trainer.optimizer, trainer.lr_scheduler
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def evaluate(self) -> None:
        torch.use_deterministic_algorithms(False)  # errorを回避
        llm = vllm.LLM(**self.cfg.vllm.model)
        # tokenizer = llm.get_tokenizer()
        sampling_params = vllm.SamplingParams(**self.cfg.vllm.sampling)
        full_responses = llm.generate(
            prompts=self.eval_dataset,
            sampling_params=sampling_params,
            lora_request=LoRARequest("adapter", 1, self.output_dir),
            use_tqdm=True,
        )
        preds = [x.outputs[0].text.replace("<|im_start|>", "") for x in full_responses]
        oof = self.valid.with_columns(pl.Series(preds).alias("pred")).select(
            ["QuestionId_Answer", "MisconceptionId", "MisconceptionName", "fold", "pred"]
        )
        oof.write_csv(self.output_dir / "oof.csv")
        # score = mapk(preds=oof["pred"].to_numpy(), labels=oof["MisconceptionId"].to_numpy())
        # LOGGER.info(f"CV: {score}")
        # wandb.log({"CV": score})  # type: ignore

    def run(self) -> None:
        self.setup_logger()
        self.setup_dataset()
        self.training()
        # self.evaluate()  # gpuがうまく解放できずoomになってしまう
        wandb.finish()  # type: ignore


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
