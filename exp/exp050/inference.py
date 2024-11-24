import gc
import re
import logging
from enum import Enum
from pathlib import Path
from collections import defaultdict

import vllm
import hydra
import numpy as np
import torch
import polars as pl
from vllm import RequestOutput
from lightning import seed_everything
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from .data_processor import TOKENIZER

LOGGER = logging.getLogger(__name__)


class LLMPredictType(Enum):
    Top1 = "top1"
    Reranking = "reranking"


class MultipleChoiceLogitsProcessor:
    """
    A logits processor to answer multiple choice questions with one of the choices.
    A multiple choice question is like:
    I am getting a lot of calls during the day. What is more important for me to consider when I buy a new phone?
    0. Camera
    1. Screen resolution
    2. Operating System
    3. Battery
    The goal is to make LLM generate "3" as an answer.


    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    choices (List[str]): List of one character answers like A, B, C, D.
    delimiter (str): One character delimiter that comes after the choices like 1. or 2-.
    boost_first_words (float): Nonzero values add choices' first tokens' logits to boost performance.
                            Especially useful for the models which have difficulty associating the choice with its text.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        choices: list[str] | None = None,
        delimiter: str = ".",
        boost_first_words: float = 0.0,
    ) -> None:
        if choices is None:
            choices = ["1", "2", "3", "4"]

        self.new_line_token = text_to_token(tokenizer, "\n", last=False)
        self.delimiter_token = text_to_token(tokenizer, delimiter, last=False)
        self.choice_tokens = [text_to_token(tokenizer, choice, last=False) for choice in choices]
        self.boost_first_words = boost_first_words
        self.very_large_number = 999

    def __call__(self, prompt_tokens_ids: list[int], past_token_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        if self.boost_first_words:
            choice = 0

            first_tokens = []
            for i in range(len(prompt_tokens_ids) - 3):
                # A choice is like "\nA) hair dryer", where first token is "hair"
                choice_starts = (
                    (prompt_tokens_ids[i] == self.new_line_token)
                    and (prompt_tokens_ids[i + 1] == self.choice_tokens[choice])
                    and (prompt_tokens_ids[i + 2] == self.delimiter_token)
                )

                if choice_starts:
                    first_tokens.append(prompt_tokens_ids[i + 3])
                    choice += 1

                    if choice >= len(self.choice_tokens):
                        break

            scores[self.choice_tokens[: len(first_tokens)]] += self.boost_first_words * scores[first_tokens]

        scores[self.choice_tokens] += self.very_large_number
        return scores


def text_to_token(tokenizer: PreTrainedTokenizer, text: str, last: bool, token_thr: int = 2) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not last and len(tokens) > token_thr:
        # Usually the first token indicates the beginning, and the second token is our main token
        raise Exception(f"Can't convert {text} to token. It has {len(tokens)} tokens.")

    return tokens[-1]


def preprocess_text(x: str) -> str:
    x = re.sub(r"http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


def add_reranking_prompt(df: pl.DataFrame, misconception: pl.DataFrame, tokenizer: TOKENIZER) -> pl.DataFrame:
    id2name_mapping = {row["MisconceptionId"]: row["MisconceptionName"] for row in misconception.iter_rows(named=True)}
    prompt = """
Below is a mathematics question on {ConstructName} ({SubjectName}):
Question: {Question}
Correct Answer: {CorrectAnswer}
Student's Incorrect Answer: {IncorrectAnswer}

As a mathematics teacher, your task is to:

- Analyze the student's incorrect answer.
- Rank the given possible misconceptions from most likely to least likely based on how they could have led to the student's error.
- Provide the ranked list of IDs separated by '\n' (newline characters) without additional explanations.
- Do not include the reasoning process.
Here are the possible misconceptions to consider (each with an ID):

{Retrieval}

Based on the above information, please rank the IDs of the misconceptions in order of likelihood that caused the student's error, separated by '\n'.
"""
    texts = [
        preprocess_text(
            prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                CorrectAnswer=row["CorrectAnswerText"],
                IncorrectAnswer=row["InCorrectAnswerText"],
                Retrieval=get_retrieval_text(row["PredictMisconceptionId"], id2name_mapping, use_misconception_id=True),
            )
        )
        for row in df.iter_rows(named=True)
    ]

    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["Prompt"]},
            ],
            # tokeadd_generation_prompt=True,
            add_generation_prompt=True,
            tokenize=False,  # textとして渡す
        )
        # + last_text
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    return df


def add_top1_prompt(df: pl.DataFrame, misconception: pl.DataFrame, tokenizer: TOKENIZER) -> pl.DataFrame:
    id2name_mapping = {row["MisconceptionId"]: row["MisconceptionName"] for row in misconception.iter_rows(named=True)}
    prompt = """
Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
Pick the correct misconception number from the below:

{Retrieval}
"""
    texts = [
        preprocess_text(
            prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                CorrectAnswer=row["CorrectAnswerText"],
                IncorrectAnswer=row["InCorrectAnswerText"],
                Retrieval=get_retrieval_text(
                    row["PredictMisconceptionId"], id2name_mapping, use_misconception_id=False
                ),
            )
        )
        for row in df.iter_rows(named=True)
    ]

    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["Prompt"]},
            ],
            # tokeadd_generation_prompt=True,
            add_generation_prompt=True,
            tokenize=False,  # textとして渡す
        )
        # + last_text
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(pl.Series(texts).alias("Prompt"))
    return df


def get_retrieval_text(
    misconception_ids: list[int], id2name_mapping: dict[int, str], use_misconception_id: bool
) -> str:
    # 並びをrandomにする
    # misconception_ids = random.sample(misconception_ids, len(misconception_ids))
    retrieval = ""
    for i, id in enumerate(misconception_ids):
        name = id2name_mapping[id]
        if use_misconception_id:
            retrieval += f"{id}: {name} \n"
        else:
            retrieval += f"{i}. {name} \n"
    return retrieval


def llm_inference(df: pl.DataFrame, misconception: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    llm = vllm.LLM(**cfg.vllm.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model.name)
    if cfg.llm_model.predict_type == LLMPredictType.Reranking.value:
        df = add_reranking_prompt(df, misconception, tokenizer)
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
        assert cfg.vllm.sampling.n == 1
        try:
            _preds = parse_inference(full_responses, cfg)
            df = df.with_columns(pl.Series(_preds).alias("PredictMisconceptionId"))
        except:
            print("Failed to parse inference")

    elif cfg.llm_model.predict_type == LLMPredictType.Top1.value:
        sampling_params = vllm.SamplingParams(
            **cfg.vllm.sampling,
            logits_processors=[
                MultipleChoiceLogitsProcessor(tokenizer, choices=[f"{i}" for i in range(cfg.max_candidates)])
            ],
        )

        indices = np.array(df["PredictMisconceptionId"].to_list())
        survivors = indices[:, -1:]  # 候補の中で最も類似度が大きいもの

        for i in range(3):
            c_indices = np.concatenate([indices[:, -8 * (i + 1) - 1 : -8 * i - 1], survivors], axis=1)
            df = add_top1_prompt(df, misconception, tokenizer)
            full_responses = llm.generate(
                prompts=df["Prompt"].to_numpy(),
                sampling_params=sampling_params,
                # lora_request=LoRARequest("adapter", 1, self.output_dir),
                use_tqdm=True,
            )
            responses = [x.outputs[0].text for x in full_responses]
            df = df.with_columns(pl.Series(responses).alias("response"))
            llm_choices = df["response"].astype(int).values - 1
            survivors = np.array([cix[best] for best, cix in zip(llm_choices, c_indices)]).reshape(-1, 1)

        results = []
        for i in range(indices.shape[0]):
            ix = indices[i]
            llm_choice = survivors[i, 0]
            results.append(" ".join([str(llm_choice)] + [str(x) for x in ix if x != llm_choice]))
        df = df.with_columns(pl.Series(results).alias("PredictMisconceptionId"))
    else:
        raise ValueError(f"Invalid predict_type: {cfg.llm_model.predict_type}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return df


def parse_inference(full_responses: list[RequestOutput], cfg: DictConfig) -> list[list[int]]:
    preds: list[list[int]] = []
    for x in full_responses:
        pred = [
            int(id)
            for id in x.outputs[0]
            .text.replace("<|im_start|>", "")
            .replace(":", "")
            .strip()
            .split("\n")[: cfg.retrieve_num]
            # idが整数に変換可能なものだけ取得
            if id.isdigit()
        ]
        preds.append(pred)
    return preds


def parse_text_inference(full_responses: list[RequestOutput]) -> list[int]:
    preds: list[int] = []
    for x in full_responses:
        preds.append(int(x.outputs[0].text.replace("<|im_start|>", "").replace(":", "").strip()))
    return preds


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
        # retrieval
        df, misconception_mapping = self.setup_dataset()
        if self.cfg.llm_model.use:
            # llm inference
            df = llm_inference(df, misconception_mapping, self.cfg)
        self.make_submission(df)


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
