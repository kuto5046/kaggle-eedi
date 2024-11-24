import gc
import os
import logging
from typing import Union, Optional
from pathlib import Path

import hydra
import numpy as np
import torch
import polars as pl
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


def encode(
    model: MODEL, tokenizer: TOKENIZER, texts: list[str], model_name: str, max_length: int = 2048, batch_size: int = 32
) -> np.ndarray:
    """
    tokenizerの設定上paddingはlongestに合わせてくれるので、max_lengthは大きめに設定しておく
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", total=len(texts) // batch_size):
        batch_texts = texts[i : i + batch_size]
        if model_name == "nvidia/NV-Embed-v2":
            embeddings = model.encode(batch_texts, max_length=32768)
        else:
            features = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**features)
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


def to_np(x: torch.tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    return x.detach().cpu().numpy()


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}"


# https://huggingface.co/BAAI/bge-en-icl
def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}\n<response>{response}"


def get_new_queries(
    queries: list[str], query_max_len: int, examples_prefix: str, tokenizer: TOKENIZER
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


def setup_lora_model(base_model: MODEL, model_name: str, lora_params: dict | None) -> MODEL:
    is_tuned_model_path = True if "exp" in model_name else False
    if is_tuned_model_path:
        lora_model = PeftModel.from_pretrained(base_model, model_name)
    else:
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
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
    return lora_model


def setup_quantized_model(base_model_name: str) -> PreTrainedModel:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModel.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        # local_files_only=False,
        trust_remote_code=True,
        # torch_dtype="auto",
        device_map="auto",
    )
    return model


def setup_model_and_tokenizer(
    base_model_name: str,
    model_name: str,
    is_quantized: bool = False,
    use_lora: bool = False,
    lora_params: Optional[dict] = None,
) -> tuple[Union[MODEL, PeftModel], TOKENIZER]:
    """
    Unified model and tokenizer setup function.
    Supports quantized and PEFT configurations.
    """
    if is_quantized:
        model = setup_quantized_model(base_model_name)
    else:
        model = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if use_lora:
        model = setup_lora_model(model, model_name, lora_params)

    return model, tokenizer


def compute_similarity(
    model: Union[MODEL, PeftModel],
    tokenizer: TOKENIZER,
    query_texts: list[str],
    passage_texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Compute similarity between queries and passages using model embeddings.
    """

    query_embs = encode(model, tokenizer, query_texts, model_name=model_name, batch_size=batch_size)
    passage_embs = encode(model, tokenizer, passage_texts, model_name=model_name, batch_size=batch_size)
    similarity = cosine_similarity(query_embs, passage_embs)
    return np.argsort(-similarity, axis=1)


def create_query_and_passage_input(
    base_model_name: str, df: pl.DataFrame, misconception_mapping: pl.DataFrame, tokenizer: TOKENIZER
) -> tuple[list[str], list[str]]:
    if base_model_name in [
        "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
        "dunzhang/stella_en_1.5B_v5",
        "nvidia/NV-Embed-v2",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct-AWQ",
    ]:
        task_description = "Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception."
        query_texts = [get_detailed_instruct(task_description, query) for query in df["AllText"].to_list()]
        passage_texts = misconception_mapping["MisconceptionName"].to_list()
    elif base_model_name in [
        "BAAI/bge-en-icl",
    ]:
        task_description = "Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception."
        query_texts = [get_detailed_instruct(task_description, query) for query in df["AllText"].to_list()]
        examples_prefix = ""  # if there not exists any examples, just set examples_prefix = ''
        query_max_len = 2048
        new_query_max_len, query_texts = get_new_queries(query_texts, query_max_len, examples_prefix, tokenizer)
        passage_texts = misconception_mapping["MisconceptionName"].to_list()
    elif base_model_name in ["Alibaba-NLP/gte-Qwen2-7B-instruct"]:
        # これでもうまくいかない
        task_description = "Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception."
        query_texts = [f"Instruct: {task_description}\nQuery: {query}" for query in df["AllText"].to_list()]
        passage_texts = misconception_mapping["MisconceptionName"].to_list()
    else:
        query_texts = df["AllText"].to_list()
        passage_texts = misconception_mapping["MisconceptionName"].to_list()
    return query_texts, passage_texts


def generate_candidates(
    df: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig, retrieval_model_name_or_path: str
) -> np.ndarray:
    base_model_name = cfg.retrieval_model.name
    use_lora = cfg.retrieval_model.use_lora
    lora_params = cfg.retrieval_model.lora
    is_quantized = cfg.retrieval_model.is_quantized
    batch_size = cfg.retrieval_model.batch_size
    use_sentence_transformers = cfg.retrieval_model.use_sentence_transformers
    if use_sentence_transformers:
        sorted_similarity = sentence_emb_similarity_by_sentence_transformers(
            retrieval_model_name_or_path, df, misconception_mapping
        )
    else:
        model, tokenizer = setup_model_and_tokenizer(
            base_model_name=base_model_name,
            model_name=retrieval_model_name_or_path,
            is_quantized=is_quantized,
            use_lora=use_lora,
            lora_params=lora_params,
        )

        query_texts, passage_texts = create_query_and_passage_input(
            base_model_name, df, misconception_mapping, tokenizer
        )
        sorted_similarity = compute_similarity(
            model,
            tokenizer,
            query_texts,
            passage_texts,
            model_name=base_model_name,
            batch_size=batch_size,
        )
    return sorted_similarity


def generate_candidates_all(
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
    cfg: DictConfig,
) -> pl.DataFrame:
    """
    Generate negative examples based on similarity calculations across different models.
    """
    model_names = cfg.retrieval_model.names
    weights = cfg.retrieval_model.weights
    max_candidates = cfg.max_candidates
    preds = []
    for retrieval_model_name in model_names:
        print(retrieval_model_name)
        sorted_similarity = generate_candidates(df, misconception_mapping, cfg, retrieval_model_name)
        preds.append(sorted_similarity[:, : max_candidates + 10])  # Collect predictions for ensemble

    # Ensemble predictions and add to the DataFrame
    pred = ensemble_predictions(preds, weights)
    df = df.with_columns(pl.Series(pred[:, :max_candidates].tolist()).alias("PredictMisconceptionId"))

    return df


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

            if self.cfg.debug:
                df = df.sample(fraction=0.05, seed=self.cfg.seed)
            # 学習用の候補を生成する
            sorted_similarity = generate_candidates(
                df, misconception, self.cfg, retrieval_model_name_or_path=self.cfg.retrieval_model.name
            )
            df = df.with_columns(
                pl.Series(sorted_similarity[:, : self.cfg.max_candidates].tolist()).alias("PredictMisconceptionId")
            )
            LOGGER.info(f"recall: {calc_recall(df):.5f}")
            LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
            df = explode_candidates(df, misconception)
            df = df.join(misconception, on="MisconceptionId", how="left")  # 正解ラベルの文字列を追加
        else:
            df = generate_candidates_all(df, misconception, self.cfg)
            df = explode_candidates(df, misconception)

        df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
