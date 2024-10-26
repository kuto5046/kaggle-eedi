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
from numpy.typing import NDArray
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

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
    """
    Question:str - MisconceptionId: int - PredictMisconceptionId: list[int]のペアのdataframeを入力とする
    """
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


def create_retrieved(df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.filter(
            pl.col("MisconceptionId").is_not_null()  # TODO: Consider ways to utilize data where MisconceptionId is NaN.
        )
        .explode("PredictMisconceptionId")
        .join(
            misconception_mapping,
            on="MisconceptionId",
        )
        .join(
            misconception_mapping.rename(lambda x: "Predict" + x),
            on="PredictMisconceptionId",
        )
    )
    return df


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
    prompt = """Here is a question about {ConstructName}({SubjectName}).
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
    # TODO: chat template使ってないや
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
    elif model_name == "Qwen/Qwen2.5-32B-Instruct-AWQ":
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
    llm = vllm.LLM(**cfg.vllm.model)
    # tokenizer = llm.get_tokenizer()
    sampling_params = vllm.SamplingParams(**cfg.vllm.sampling)
    full_responses = llm.generate(
        prompts=df["Prompt"].to_numpy(),
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, cfg.lora_model_path),
        use_tqdm=True,
    )

    # question,idxをkeyとしてmisconception_idを取得する
    candidates = defaultdict(list)
    for row in df.iter_rows(named=True):
        candidates[row["QuestionId"]] = row["PredictMisconceptionId"]

    preds = [x.outputs[0].text.replace("<|im_start|>", "").replace(":", "").strip() for x in full_responses]
    df = df.with_columns(pl.Series(preds).alias("LLMPredictMisconceptionName")).with_columns(
        pl.concat_str(
            [
                pl.col("ConstructName"),
                pl.col("SubjectName"),
                pl.col("QuestionText"),
                pl.col("InCorrectAnswerText"),
                pl.col("LLMPredictMisconceptionName"),
            ],
            separator=" ",
        ).alias("AllText")
    )
    return df


def generate_candidates(
    df: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig, num_candidates: int
) -> pl.DataFrame:
    # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
    model_dir = str(cfg.path.model_dir / cfg.retrieval_model.pretrained_exp_name / f"fold{cfg.use_fold}")
    model = SentenceTransformer(model_dir)
    sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model)
    df = df.with_columns(
        pl.Series(sorted_similarity[:, :num_candidates].tolist()).alias("PredictMisconceptionId")
    ).filter(pl.col("MisconceptionId").is_not_null())
    return df


class DataProcessor:
    def __init__(self, cfg: DictConfig) -> None:
        for key, value in cfg.path.items():
            cfg.path[key] = Path(value)
        if cfg.debug:
            cfg.run_name = "debug"
        self.output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        seed_everything(cfg.seed, workers=True)  # data loaderのworkerもseedする
        self.cfg = cfg
        self.common_cols = ["QuestionId", "ConstructName", "SubjectName", "QuestionText", "CorrectAnswer"]

    def read_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = pl.read_csv(self.cfg.path.input_dir / f"{self.cfg.phase}.csv")
        misconception_mapping = pl.read_csv(self.cfg.path.input_dir / "misconception_mapping.csv")
        return df, misconception_mapping

    def preprocess(self, input_df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = preprocess_table(input_df, self.common_cols)
        # misconception情報(target)を取得
        pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
        df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
        df = df.filter(pl.col("MisconceptionId").is_not_null())
        df = df.join(misconception, on="MisconceptionId", how="left")
        return df

    def add_fold(self, df: pl.DataFrame) -> pl.DataFrame:
        return get_groupkfold(df, group_col="QuestionId", n_splits=self.cfg.n_splits)

    def feature_engineering(self, df: pl.DataFrame, misconception: pl.DataFrame) -> pl.DataFrame:
        df = generate_candidates(df, misconception, self.cfg, num_candidates=self.cfg.max_candidates)
        df = add_prompt(df, misconception, self.cfg.llm_model.name)
        # LLMで予測
        df = llm_inference(df, self.cfg)
        df.select(
            ["QuestionId_Answer", "MisconceptionId", "MisconceptionName", "LLMPredictMisconceptionName"]
        ).write_csv(self.output_dir / "eval.csv")
        df = generate_candidates(df, misconception, self.cfg, num_candidates=self.cfg.retrieve_num)
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df, misconception)
        df = self.add_fold(df)
        use_df = df.filter(pl.col("fold") == self.cfg.use_fold).sample(n=100, shuffle=True, seed=self.cfg.seed)
        use_df = self.feature_engineering(use_df, misconception)
        LOGGER.info(f"fold={self.cfg.use_fold} validation size: {len(use_df)}")
        LOGGER.info(f"recall: {calc_recall(use_df):.5f}")
        LOGGER.info(f"mapk: {calc_mapk(use_df):.5f}")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
