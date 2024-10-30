import re
import logging
from pathlib import Path
from collections import defaultdict

import vllm
import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .data_processor import generate_candidates

LOGGER = logging.getLogger(__name__)


def preprocess_text(x: str) -> str:
    x = re.sub(r"http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


def add_prompt(df: pl.DataFrame, misconception: pl.DataFrame, model_name: str) -> pl.DataFrame:
    prompt = """You are a Mathematics teacher with the task of evaluating a student's understanding.
The student has answered a question about {ConstructName} ({SubjectName}) incorrectly.
Your task is to carefully analyze and reason through the question and the student's incorrect answer
to determine whether the given Misconception accurately describes the student's misunderstanding.

Here is the detailed information:

- **Question**: {Question}
- **Correct Answer**: {CorrectAnswer}
- **Incorrect Answer**: {IncorrectAnswer}

You will evaluate whether the provided Misconception truly captures the reason behind the Incorrect Answer.
If the Misconception fully explains why the student might have chosen the Incorrect Answer, reply with "Yes."
If the Misconception does not explain the student's reasoning or if another reason better explains the incorrect answer,
reply with "No."

**Misconception**: {MisconceptionName}

Please focus on the following aspects in your analysis:
1. Does the Incorrect Answer logically follow from the Misconception, given the context of the Question?
2. Does the Misconception specifically relate to the key concepts in {ConstructName} ({SubjectName}) relevant to this question?
3. Could this Misconception reasonably cause confusion that would lead to the Incorrect Answer?

Be precise and concise in your response, responding only with "Yes" or "No" based on your analysis.
"""
    # id2name_mapping = {row["MisconceptionId"]: row["MisconceptionName"] for row in misconception.iter_rows(named=True)}
    texts = [
        preprocess_text(
            prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                CorrectAnswer=row["CorrectAnswerText"],
                IncorrectAnswer=row["InCorrectAnswerText"],
                MisconceptionName=row["MisconceptionName"],
            )
        )
        for row in df.iter_rows(named=True)
    ]
    if model_name == "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4":
        last_text = "<|start_header_id|>assistant<|end_header_id|>"
    elif model_name in ["Qwen/Qwen2.5-32B-Instruct-AWQ", "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"]:
        last_text = "<|im_start|>assistant"
    else:
        last_text = ""
    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["Prompt"]},
            ],
            tokeadd_generation_prompt=True,
            tokenize=False,  # textとして渡す
        )
        + last_text
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    return df


def get_retrieval_text(misconception_ids: list[int], id2name_mapping: dict[int, str]) -> str:
    retrieval = ""
    for i, id in enumerate(misconception_ids):
        name = id2name_mapping[id]
        retrieval += f"- {name} \n"
    return retrieval


def llm_inference(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    if cfg.llm_model.name in ["Qwen/Qwen2.5-Math-7B-Instruct"]:
        # https://docs.vllm.ai/en/v0.6.0/quantization/auto_awq.html
        # quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        # quant_path = str(cfg.path.output_dir)
        # # Load model
        # model = AutoAWQForCausalLM.from_pretrained(
        #     cfg.llm_model.name,
        #     device_map="cpu",
        #     **{"low_cpu_mem_usage": True, "use_cache": False}
        # )
        # tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model.name, trust_remote_code=True).to("cpu")

        # # Quantize
        # model.quantize(tokenizer, quant_config=quant_config)

        # # Save quantized model
        # model.save_quantized(quant_path)
        # tokenizer.save_pretrained(quant_path)
        # cfg.vllm.model.model = quant_path
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

    # question,idxをkeyとしてmisconception_idを取得する
    candidates = defaultdict(list)
    for row in df.iter_rows(named=True):
        candidates[row["QuestionId"]] = row["PredictMisconceptionId"]

    preds = []
    for x in full_responses:
        probs = []
        for output in x.outputs:
            responce = output.text.replace("<|im_start|>", "").replace(":", "").strip() + " "
            prob = np.exp(next(iter(x.outputs[0].logprobs[0].values())).logprob)
            if responce == "Yes":
                pass
            elif responce == "No":
                prob = 1 - prob
            else:
                ValueError
            probs.append(prob)  # 1である確率を返す
        pred = np.mean(probs)
        preds.append(pred)
    df = df.with_columns(pl.Series(preds).alias("prob"))
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
        # llm inference
        df = add_prompt(df, misconception_mapping, self.cfg.llm_model.name)
        df = llm_inference(df, self.cfg)
        # second retreval
        df = generate_candidates(df, misconception_mapping, self.cfg.retrieval_model.names, self.cfg.retrieve_num)
        self.make_submission(df)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
