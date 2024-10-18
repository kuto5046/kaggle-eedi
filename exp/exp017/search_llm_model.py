import re
import logging
from pathlib import Path

import vllm
import hydra
import scipy
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

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
            #     pl.concat_str(
            #         [
            #             pl.col("ConstructName"),
            #             pl.col("SubjectName"),
            #             pl.col("QuestionText"),
            #             pl.col("AnswerText"),
            #         ],
            #         separator=" ",
            #     ).alias("AllText"),
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
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


def preprocess(input_df: pl.DataFrame) -> pl.DataFrame:
    common_cols = ["QuestionId", "ConstructName", "SubjectName", "QuestionText", "CorrectAnswer"]
    df = preprocess_table(input_df, common_cols)
    # misconception情報(target)を取得
    pp_misconception_mapping = preprocess_misconception(input_df, common_cols)
    df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
    df = df.filter(pl.col("MisconceptionId").is_not_null())
    return df


def add_llm_misconception_features(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    # プロンプト生成
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model.model)
    prompt = """
    Question: {Question}
    Incorrect Answer: {IncorrectAnswer}
    Correct Answer: {CorrectAnswer}
    Construct Name: {ConstructName}
    Subject Name: {SubjectName}

    Your task: Identify the misconception behind Incorrect Answer.
    Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
    Before answering the question think step by step concisely in 1-2 sentence
    inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and
    respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag.
    """
    texts = [apply_template(row, tokenizer, prompt) for row in df.iter_rows(named=True)]
    df = df.with_columns(pl.Series(texts).alias("Prompt"))

    # LLMによるmisconception予測
    llm = vllm.LLM(**cfg.llm.model)
    tokenizer = llm.get_tokenizer()
    prompts = df["Prompt"].to_numpy()
    sampling_params = vllm.SamplingParams(**cfg.llm.sampling)
    full_responses = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)

    def extract_response(text: str) -> str:
        return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

    responses = [extract_response(x.outputs[0].text) for x in full_responses]
    df = df.with_columns(pl.Series(responses).alias("LLMMisconception"))
    df = df.drop("Prompt")
    # df = df.with_columns(
    #     pl.concat_str(
    #         [
    #             pl.col("ConstructName"),
    #             pl.col("SubjectName"),
    #             pl.col("QuestionText"),
    #             pl.col("AnswerText"),
    #             pl.col("LLMMisconception"),
    #         ],
    #         separator=" ",
    #     ).alias("AllText")
    # )
    return df


def apply_template(row: pl.Series, tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                IncorrectAnswer=row["CorrectAnswer"],
                CorrectAnswer=row["AnswerText"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig):
    # 評価対象のモデル
    # ref: https://docs.vllm.ai/en/latest/models/supported_models.html
    seed_everything(cfg.seed)
    for key, value in cfg.path.items():
        cfg.path[key] = Path(value)

    for key, value in cfg.llm.model.items():
        LOGGER.info(f"{key}: {value}")

    output_dir = cfg.path.output_dir / cfg.exp_name / cfg.run_name
    output_dir.mkdir(exist_ok=True, parents=True)

    train = pl.read_csv(cfg.path.input_dir / "train.csv")
    misconception_mapping = pl.read_csv(cfg.path.input_dir / "misconception_mapping.csv")
    df = preprocess(train)

    emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    df = add_llm_misconception_features(df, cfg)
    model_name = cfg.llm.model.model.replace("/", "-")
    df.write_csv(output_dir / f"{model_name}.csv")
    df = df.join(misconception_mapping, on="MisconceptionId", how="left")

    gt_emb = emb_model.encode(df["MisconceptionName"].to_list(), normalize_embeddings=True)
    pred_emb = emb_model.encode(df["LLMMisconception"].to_list(), normalize_embeddings=True)

    # 値が小さいほど類似度が大きい
    d = scipy.spatial.distance.cosine(gt_emb.flatten(), pred_emb.flatten())
    LOGGER.info(f"{model_name}: {d:.5f}")

    # similarity = cosine_similarity(gt_emb, pred_emb)
    # similarity = similarity * np.eye(similarity.shape[0])
    # 1 - (similarity[similarity > 0]).mean()


if __name__ == "__main__":
    main()
