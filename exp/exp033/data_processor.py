import gc
import logging
from pathlib import Path
import os
import hydra
import numpy as np
import torch
import polars as pl
import torch.nn.functional as F
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = logging.getLogger(__name__)


def preprocess_table(df: pl.DataFrame, common_cols: list[str]) -> pl.DataFrame:
    long_df = (
        df.select(pl.col(common_cols + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_cols,
            variable_name="AnswerType",
            value_name="AnswerText",
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("ConstructName"),
                    pl.col("SubjectName"),
                    pl.col("QuestionText"),
                    pl.col("AnswerText"),
                ],
                separator=" ",
            ).alias("AllText"),
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
    df2 = df.explode("PredictMisconceptionId")
    return (
        df2.filter(pl.col("MisconceptionId") == pl.col("PredictMisconceptionId"))["QuestionId_Answer"].n_unique()
        / df2["QuestionId_Answer"].n_unique()
    )


# ref: https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1#MAP@3-Metric
def mapk(preds: NDArray[np.object_], labels: NDArray[np.int_], k: int = 25) -> float:
    map_sum = 0
    for _x, y in zip(preds, labels):
        x = [int(i) for i in _x.split(" ")]
        z = [1 / i if y == j else 0 for i, j in zip(range(1, k + 1), x)]
        map_sum += np.sum(z)
    return map_sum / len(preds)


def calc_mapk(df: pl.DataFrame) -> float:
    """
    Question:str - MisconceptionId: int - PredictMisconceptionId: list[int]のペアのdataframeを入力とする
    """
    agg_df = (
        df.with_columns(
            pl.col("PredictMisconceptionId").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String)
        )
        .join(
            df[["QuestionId_Answer", "MisconceptionId"]].unique(),
            on=["QuestionId_Answer"],
        )
        .sort(by=["QuestionId_Answer"])
    )
    return mapk(agg_df["PredictMisconceptionId"].to_numpy(), agg_df["MisconceptionId"])


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


def sentence_emb_similarity(
    df: pl.DataFrame,
    misconception_mapping: pl.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    text_vec = model.encode(df["AllText"].to_list(), normalize_embeddings=True)
    misconception_mapping_vec = model.encode(
        misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
    )
    similarity = cosine_similarity(text_vec, misconception_mapping_vec)
    sorted_similarity = np.argsort(-similarity, axis=1)
    return sorted_similarity


def explode_candidates(df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
    df = df.explode("PredictMisconceptionId").join(
        misconception_mapping.rename(lambda x: "Predict" + x),
        on="PredictMisconceptionId",
    )
    return df


def add_eos(eos_token: str, input_examples: list[str]) -> list[str]:
    input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples


def to_np(x: torch.tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def last_token_pool(last_hidden_states: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


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
    retrieval_model_names: list[str | Path],
    num_candidates: int,
    weights: list[float] | None = None,
    local_files_only: bool = False,
) -> pl.DataFrame:
    # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
    preds = []
    for retrieval_model_name in retrieval_model_names:
        print(str(retrieval_model_name))
        if retrieval_model_name in ["nvidia/NV-Embed-v2"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModel.from_pretrained(
                retrieval_model_name, quantization_config=bnb_config, trust_remote_code=True
            )

            max_length = 1024  # 32768
            batch_size = 64
            num_workers = 12
            # get the embeddings
            instruct = "Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception."
            query_prefix = f"Instruct: {instruct} \nQuery: "
            passage_prefix = ""

            query_embeddings = model._do_encode(
                df["AllText"].to_list(), instruction=query_prefix, max_length=max_length, num_workers=num_workers, batch_size=batch_size
            )
            passage_embeddings = model._do_encode(
                misconception_mapping["MisconceptionName"].to_list(), instruction=passage_prefix, max_length=max_length, num_workers=num_workers, batch_size=batch_size
            )

            # normalize embeddings
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

            similarity = to_np(query_embeddings @ passage_embeddings.T)
            sorted_similarity = np.argsort(-similarity, axis=1)
        else:
            model = SentenceTransformer(
                str(retrieval_model_name), local_files_only=local_files_only, trust_remote_code=True
            )
            sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model)
        preds.append(sorted_similarity[:, : num_candidates + 10])  # アンサンブル用に大きめに計算

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    pred = ensemble_predictions(preds, weights)
    df = df.with_columns(pl.Series(pred[:, :num_candidates].tolist()).alias("PredictMisconceptionId"))

    return df


# https://www.kaggle.com/code/titericz/h-m-ensembling-how-to
def ensemble_predictions(preds: list[np.ndarray], weights: list[float] | None = None, top_k: int = 25) -> np.ndarray:
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
        result = list(dict(sorted(scores.items(), key=lambda item: -item[1])).keys())[:top_k]
        blend_results.append(result)
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
        return get_groupkfold(df, group_col="QuestionId", n_splits=self.cfg.n_splits)

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
            df = generate_candidates(df, misconception, [self.cfg.retrieval_model.name], self.cfg.max_candidates)
            LOGGER.info(f"recall: {calc_recall(df):.5f}")
            LOGGER.info(f"mapk: {calc_mapk(df):.5f}")
            df = explode_candidates(df, misconception)
            df = df.join(misconception, on="MisconceptionId", how="left")  # 正解ラベルの文字列を追加
        else:
            df = generate_candidates(
                df,
                misconception,
                self.cfg.retrieval_model.names,
                self.cfg.max_candidates,
                self.cfg.retrieval_model.weights,
                local_files_only=True,
            )
            df = explode_candidates(df, misconception)

        df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
