import gc
import os
import re
import logging
from typing import Any, Union, Optional
from pathlib import Path

import hydra
import numpy as np
import torch
import polars as pl
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
from torch import nn
from lightning import seed_everything
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = logging.getLogger(__name__)


TOKENIZER = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
MODEL = Union[PreTrainedModel, SentenceTransformer, nn.Module]


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
    prompt = """
Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher.
Your task is to reason and identify the misconception behind the Incorrect Answer with the Question in English.
Answer concisely what misconception it is to lead to getting the incorrect answer.
No need to give the reasoning process and do not use "The misconception is" to start your answers.
There are some relative and possible misconceptions below to help you make the decision:

{Retrieval}
"""
    id2name_mapping = {row["MisconceptionId"]: row["MisconceptionName"] for row in misconception.iter_rows(named=True)}
    texts = [
        preprocess_text(
            prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                CorrectAnswer=row["CorrectAnswerText"],
                IncorrectAnswer=row["InCorrectAnswerText"],
                Retrieval=get_retrieval_text(row["PredictMisconceptionId"], id2name_mapping),
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


def encode(
    model: MODEL, tokenizer: TOKENIZER, texts: list[str], model_name: str, max_length: int = 2048, batch_size: int = 32
) -> np.ndarray:
    """
    tokenizerの設定上paddingはlongestに合わせてくれるので、max_lengthは大きめに設定しておく
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", total=len(texts) // batch_size):
        batch_texts = texts[i : i + batch_size]
        features = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            outputs = model(**features)
            if model_name == "nvidia/NV-Embed-v2":
                embeddings = last_token_pool(outputs["sentence_embeddings"], features["attention_mask"])
            else:
                embeddings = last_token_pool(outputs.last_hidden_state, features["attention_mask"])
            norm_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        all_embeddings.append(to_np(norm_embeddings))
    return np.vstack(all_embeddings)


def last_token_pool(last_hidden_states: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def preprocess_table(df: pl.DataFrame, common_cols: list[str]) -> pl.DataFrame:
    long_df = (
        df.select(pl.col(common_cols + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_cols,
            variable_name="AnswerType",
            value_name="AnswerText",
        )
        .with_columns(
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
    )
    # 問題-正解-不正解のペアを作る
    correct_df = (
        long_df.filter(pl.col("CorrectAnswer") == pl.col("AnswerAlphabet"))
        .select(["QuestionId", "AnswerAlphabet", "AnswerText"])
        .rename({"AnswerAlphabet": "CorrectAnswerAlphabet", "AnswerText": "CorrectAnswerText"})
    )
    long_df = (
        long_df.join(correct_df, on=["QuestionId"], how="left")
        .rename({"AnswerAlphabet": "InCorrectAnswerAlphabet", "AnswerText": "InCorrectAnswerText"})
        .filter(pl.col("InCorrectAnswerAlphabet") != pl.col("CorrectAnswerAlphabet"))
        .drop(["AnswerType", "CorrectAnswer"])
    )
    long_df = long_df.with_columns(
        pl.concat_str(
            [
                pl.lit("\n## Construct\n"),
                pl.col("ConstructName"),
                pl.lit("\n## Subject\n"),
                pl.col("SubjectName"),
                pl.lit("\n## Question\n"),
                pl.col("QuestionText"),
                pl.lit("\n## CorrectAnswer\n"),
                pl.col("CorrectAnswerText"),
                pl.lit("\n## InCorrectAnswer\n"),
                pl.col("InCorrectAnswerText"),
            ],
            separator="",
        ).alias("AllText")
    )
    return long_df


def preprocess_misconception(df: pl.DataFrame, common_cols: list[str]) -> pl.DataFrame:
    misconception = (
        df.select(pl.col(common_cols + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_cols,
            variable_name="MisconceptionType",
            value_name="MisconceptionId",
        )
        .with_columns(
            pl.col("MisconceptionType").str.extract(r"Misconception([A-D])Id$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
        .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )
    return misconception


def calc_recall(df: pl.DataFrame) -> float:
    df2 = df.with_columns(pl.col("PredictMisconceptionId").list.slice(0, 25).alias("PredictMisconceptionId"))
    df2 = df2.explode("PredictMisconceptionId")
    return (
        df2.filter(pl.col("MisconceptionId") == pl.col("PredictMisconceptionId"))["QuestionId_Answer"].n_unique()
        / df2["QuestionId_Answer"].n_unique()
    )


def apk(actual: list[int], predicted: list[int], k: int = 25) -> float:
    """Computes the average precision at k.

    Parameters
    ----------
    actual : A list of elements that are to be predicted (order doesn't matter)
    predicted : A list of predicted elements (order does matter)

    Returns
    -------
    score : The average precision at k over the input lists
    """

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted[:k]):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual: list[list[int]], predicted: list[list[int]], k: int = 25) -> float:
    """Computes the mean average precision at k.

    Parameters
    ----------
    actual : A list of lists of elements that are to be predicted (order doesn't matter)
    predicted : list of lists of predicted elements (order matters in the lists)
    k : The maximum number of predicted elements

    Returns
    -------
    score : The mean average precision at k over the input lists
    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]).item()


def calc_mapk(df: pl.DataFrame) -> float:
    """
    Question:str - MisconceptionId: int - PredictMisconceptionId: list[int]のペアのdataframeを入力とする
    """
    # 正解データもリスト形式にする
    agg_df = (
        df.drop(["MisconceptionId"])
        .join(
            df.group_by({"QuestionId_Answer"}).agg(["MisconceptionId"]),
            on=["QuestionId_Answer"],
        )
        .sort(by=["QuestionId_Answer"])
    )
    return mapk(agg_df["MisconceptionId"].to_list(), agg_df["PredictMisconceptionId"].to_list())


def get_fold(_train: pl.DataFrame, cv: list[tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    """
    trainにfoldのcolumnを付与する
    """
    train = _train.clone()
    train = train.with_columns(pl.lit(-1).alias("fold"))
    for fold, (train_idx, valid_idx) in enumerate(cv):
        train = train.with_columns(
            pl.when(pl.arange(0, len(train)).is_in(valid_idx)).then(fold).otherwise(pl.col("fold")).alias("fold")
        )
    LOGGER.info(train.group_by("fold").len().sort("fold"))
    return train


def get_groupkfold(train: pl.DataFrame, group_col: str, n_splits: int) -> pl.DataFrame:
    kf = GroupKFold(n_splits=n_splits)
    cv = list(kf.split(X=train, groups=train[group_col].to_numpy()))
    return get_fold(train, cv)


def get_stratifiedkfold(train: pl.DataFrame, target_col: str, n_splits: int, seed: int) -> pl.DataFrame:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train, y=train[target_col].to_numpy()))
    return get_fold(train, cv)


def setup_quantized_model(model_name: str) -> tuple[MODEL, TOKENIZER]:
    if model_name == "Qwen/Qwen2.5-32B-Instruct-AWQ":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # quantization_config=quantization_config,
            use_cache=False,
            # attn_implementation="flash_attention_2",  # メモリ節約目的だが変わらず
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # 量子化したモデルを読み込む
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            # use_cache=False,  # nvidiaモデルはエラーが出る。コメントアウトするとgradient checkpointingの際にwarningが出るが問題はない
            local_files_only=False,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return base_model, tokenizer


def setup_qlora_model(
    base_model: MODEL, lora_params: dict[str, Any], pretrained_lora_path: Optional[str | Path]
) -> tuple[PeftModel, TOKENIZER]:
    if pretrained_lora_path is None:
        # LoRAアダプタを追加
        lora_config = LoraConfig(
            **lora_params,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="DEFAULT",
        )
        base_model.enable_input_require_grads()
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
    else:
        lora_model = PeftModel.from_pretrained(base_model, str(pretrained_lora_path))
    return lora_model


def sentence_emb_similarity_by_sentence_transformers(
    retrieval_model_name: str | Path,
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
) -> np.ndarray:
    model = SentenceTransformer(str(retrieval_model_name), local_files_only=False, trust_remote_code=True)
    text_vec = model.encode(df["AllText"].to_list(), normalize_embeddings=True)
    misconception_mapping_vec = model.encode(
        misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
    )
    similarity = cosine_similarity(text_vec, misconception_mapping_vec)
    sorted_similarity = np.argsort(-similarity, axis=1)
    del model
    clean_gpu()
    return sorted_similarity


def sentence_emb_similarity_by_peft(
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
    model_name: str,
    lora_params: dict[str, Any],
    batch_size: int,
    pretrained_lora_path: Optional[str | Path],
) -> np.ndarray:
    base_model, tokenizer = setup_quantized_model(model_name)
    model = setup_qlora_model(base_model, lora_params, pretrained_lora_path)
    query_embs = encode(
        model,
        tokenizer,
        df["AllText"].to_list(),
        model_name=model_name,
        batch_size=batch_size,
    )
    passage_embs = encode(
        model,
        tokenizer,
        misconception_mapping["MisconceptionName"].to_list(),
        model_name=model_name,
        batch_size=batch_size,
    )
    similarity = cosine_similarity(query_embs, passage_embs)
    sorted_similarity = np.argsort(-similarity, axis=1)
    del model, tokenizer
    clean_gpu()
    return sorted_similarity


def sentence_emb_similarity_by_nvidia(
    retrieval_model_name: str, df: pl.DataFrame, misconception_mapping: pl.DataFrame
) -> np.ndarray:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(retrieval_model_name, quantization_config=bnb_config, trust_remote_code=True)

    max_length = 32768  # longestに合わせてくれるので大きい値を入れる
    batch_size = 32
    num_workers = min(os.cpu_count(), 12)  # type: ignore
    # get the embeddings
    instruct = "Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception."
    query_prefix = f"Instruct: {instruct} \nQuery: "
    df = df.with_columns([(pl.lit(query_prefix) + pl.col("AllText")).alias("AllText")])

    query_embeddings = model._do_encode(
        df["AllText"].to_list(),
        instruction="",
        max_length=max_length,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    passage_embeddings = model._do_encode(
        misconception_mapping["MisconceptionName"].to_list(),
        instruction="",
        max_length=max_length,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    # normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

    similarity = to_np(query_embeddings @ passage_embeddings.T)
    sorted_similarity = np.argsort(-similarity, axis=1)
    del model
    clean_gpu()
    return sorted_similarity


def clean_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def explode_candidates(df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
    df = df.explode("PredictMisconceptionId").join(
        misconception_mapping.rename(lambda x: "Predict" + x),
        on="PredictMisconceptionId",
        how="left",  # inner joinだとpredict idがsortされてしまう
    )
    return df


def add_eos(eos_token: str, input_examples: list[str]) -> list[str]:
    input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples


def to_np(x: torch.tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}"


def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}\n<response>{response}"


def get_new_queries(
    queries: list[str], query_max_len: int, examples_prefix: str, tokenizer: AutoTokenizer
) -> tuple[int, list[str]]:
    inputs = tokenizer(
        queries,
        max_length=query_max_len
        - len(tokenizer("<s>", add_special_tokens=False)["input_ids"])
        - len(tokenizer("\n<response></s>", add_special_tokens=False)["input_ids"]),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer("\n<response>", add_special_tokens=False)["input_ids"]
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs["input_ids"])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + "\n<response>"
    return new_max_length, new_queries


def generate_candidates(
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
    cfg: DictConfig,
) -> pl.DataFrame:
    # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
    oofs = []
    for fold in range(cfg.n_splits):
        oof = df.filter(pl.col("fold") == fold).clone()
        sorted_similarity = sentence_emb_similarity_by_peft(
            oof,
            misconception_mapping,
            cfg.retrieval_model.name,
            cfg.retrieval_model.lora,
            cfg.trainer.batch_size,
            pretrained_lora_path=Path(cfg.retrieval_model.pretrained_path) / f"run{fold}",
        )
        oofs.append(
            oof.with_columns(
                pl.Series(sorted_similarity[:, : cfg.max_candidates].tolist()).alias("PredictMisconceptionId")
            )
        )
    output_df = pl.concat(oofs).filter(pl.col("MisconceptionId").is_not_null()).sort("QuestionId_Answer")
    return output_df


# https://www.kaggle.com/code/titericz/h-m-ensembling-how-to
def ensemble_predictions(preds: list[np.ndarray], weights: list[float] | None = None, top_k: int = 30) -> np.ndarray:
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
        result = list(dict(sorted(scores.items(), key=lambda item: -item[1])).keys())  # [:top_k]
        blend_results.append(result[:top_k])
    return np.array(blend_results)


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)

        self.output_dir = cfg.path.feature_dir / cfg.feature_version
        self.output_dir.mkdir(exist_ok=True, parents=True)

        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.common_cols = ["QuestionId", "ConstructName", "SubjectName", "QuestionText", "CorrectAnswer"]

    def read_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.read_csv(self.cfg.path.input_dir / f"{self.cfg.phase}.csv")
        misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")
        return df, misconception_mapping

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

    def run(self) -> None:
        input_df, misconception = self.read_data()
        df = preprocess_table(input_df, self.common_cols)
        if self.cfg.phase == "train":
            # misconception情報(target)を取得
            pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
            df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
            df = df.filter(pl.col("MisconceptionId").is_not_null())
            df = self.add_fold(df)
            # 学習用の候補を生成する
            df = generate_candidates(df, misconception, self.cfg)
            LOGGER.info(f"recall: {calc_recall(df):.5f}")
            LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
            # df = explode_candidates(df, misconception)
            df = df.join(misconception, on="MisconceptionId", how="left")  # 正解ラベルの文字列を追加
            df = add_prompt(df, misconception, self.cfg.llm_model.name)
        else:
            df = generate_candidates(df, misconception, self.cfg)
            df = explode_candidates(df, misconception)
            df = add_prompt(df, misconception, self.cfg.llm_model.name)

        df.select(["QuestionId_Answer", "MisconceptionId", "MisconceptionName", "fold", "Prompt"]).write_csv(
            self.output_dir / f"{self.cfg.phase}.csv"
        )


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
