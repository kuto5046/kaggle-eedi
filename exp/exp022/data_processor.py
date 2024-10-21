import re
import logging
from pathlib import Path

import vllm
import hydra
import numpy as np
import polars as pl
from lightning import seed_everything
from omegaconf import DictConfig
from numpy.typing import NDArray
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

# trainデータに含まれるmisconceptionのユニーク数
ORIGINAL_TRAIN_UNIQUE_MISCONCEPTION_SIZE = 1604


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
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("ConstructName"),
                pl.col("SubjectName"),
                pl.col("QuestionText"),
                pl.col("AnswerText"),
                pl.col("LLMMisconception"),
            ],
            separator=" ",
        ).alias("AllText")
    )
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
    return (
        df.filter(pl.col("MisconceptionId") == pl.col("PredictMisconceptionId"))["QuestionId_Answer"].n_unique()
        / df["QuestionId_Answer"].n_unique()
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
    agg_df = (
        df.group_by(["QuestionId_Answer"], maintain_order=True)
        .agg(pl.col("PredictMisconceptionId").alias("Predict"))
        .with_columns(pl.col("Predict").map_elements(lambda x: " ".join(map(str, x)), return_dtype=pl.String))
        .join(
            df[["QuestionId_Answer", "MisconceptionId"]].unique(),
            on=["QuestionId_Answer"],
        )
        .sort(by=["QuestionId_Answer"])
    )
    return mapk(agg_df["Predict"].to_numpy(), agg_df["MisconceptionId"])


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


def get_groupkfold(train: pl.DataFrame, target_col: str, n_splits: int) -> pl.DataFrame:
    kf = GroupKFold(n_splits=n_splits)
    cv = list(kf.split(X=train, groups=train[target_col].to_numpy()))
    return get_fold(train, cv)


def get_stratifiedgroupkfold(
    train: pl.DataFrame, target_col: str, group_col: str, n_splits: int, seed: int
) -> pl.DataFrame:
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train, y=train[target_col].to_numpy(), groups=train[group_col].to_numpy()))
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


def create_retrieved(df: pl.DataFrame, misconception_mapping: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
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
        .with_row_index(name="idx")
    )

    tmp = (
        df.with_columns((pl.col("MisconceptionId") == pl.col("PredictMisconceptionId")).alias("is_gt_candidate"))
        .group_by("QuestionId_Answer")
        .agg(pl.col("is_gt_candidate").sum())
    )
    gt_no_ids = tmp.filter(pl.col("is_gt_candidate") == 0.0)["QuestionId_Answer"].to_list()
    gt_has_ids = tmp.filter(pl.col("is_gt_candidate") > 0.0)["QuestionId_Answer"].to_list()
    gt_no_df = df.filter(pl.col("QuestionId_Answer").is_in(gt_no_ids))
    gt_has_df = df.filter(pl.col("QuestionId_Answer").is_in(gt_has_ids))
    # 学習前の時点で正しい予測より類似度が低いものに関しては学習データから除外する
    gt_has_df = (
        gt_has_df.with_columns(
            [
                # MisconceptionIdごとに0から始まるrankを付ける
                pl.col("PredictMisconceptionId").rank().over("QuestionId_Answer").alias("rank"),
                # MisconceptionId内でMisconceptionId=PredictMisconceptionIdとなるrankを計算
                pl.when(pl.col("MisconceptionId") == pl.col("PredictMisconceptionId"))
                .then(pl.col("PredictMisconceptionId").rank().over("QuestionId_Answer"))
                .otherwise(None)
                .alias("threshold_rank"),
            ]
        )
        .with_columns(pl.col("threshold_rank").max().over("QuestionId_Answer").alias("max_threshold_rank"))
        .filter(pl.col("rank") <= pl.col("max_threshold_rank") + cfg.tolerance_negative)
        .drop(["rank", "threshold_rank", "max_threshold_rank"])
    )
    output_df = pl.concat([gt_has_df, gt_no_df]).sort(["idx"]).drop("idx")
    return output_df


def adjust_unseen(
    seen: pl.DataFrame, unseen: pl.DataFrame, fold: int, unseen_valid_rate: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = seen.filter(pl.col("fold") != fold).drop("fold")
    valid = seen.filter(pl.col("fold") == fold).drop("fold")
    unseen_valid = unseen.sample(fraction=unseen_valid_rate, seed=seed)
    unseen_train = unseen.filter(~pl.col("QuestionId").is_in(unseen_valid["QuestionId"].to_list()))
    train = pl.concat([train, unseen_train])
    valid = pl.concat([valid, unseen_valid])
    unseen_rate = calcuate_unseen_rate(train, valid)
    valid_rate = len(valid) / (len(train) + len(valid))
    train_misconception_rate = len(train["MisconceptionId"].unique()) / ORIGINAL_TRAIN_UNIQUE_MISCONCEPTION_SIZE
    LOGGER.info(f"{fold=}: {unseen_rate=:.3%}, {valid_rate=:.3%} {train_misconception_rate=:.3%}")
    return train, valid


def calcuate_unseen_rate(train: pl.DataFrame, valid: pl.DataFrame) -> float:
    train_misconception_ids = train["MisconceptionId"].to_list()
    valid_misconception_ids = valid["MisconceptionId"].to_list()
    # unseen rateを計算
    unseen_misconception_ids = list(set(valid_misconception_ids) - set(train_misconception_ids))
    unseen_rate = len(unseen_misconception_ids) / len(valid_misconception_ids)
    return unseen_rate


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

    def preprocess(self, input_df: pl.DataFrame) -> pl.DataFrame:
        df = preprocess_table(input_df, self.common_cols)
        if self.cfg.phase == "test":
            return df
        else:
            # misconception情報(target)を取得
            pp_misconception_mapping = preprocess_misconception(input_df, self.common_cols)
            df = df.join(pp_misconception_mapping, on="QuestionId_Answer", how="inner")
            df = df.filter(pl.col("MisconceptionId").is_not_null())
            return df

    def add_fold(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        # trainデータに1度しか出現していないmisconceptionIdを抽出し、unseenとして分ける
        candidate_misconception_ids = (
            df.group_by("MisconceptionId").len().filter(pl.col("len") == 1)["MisconceptionId"].to_list()
        )
        # 未知のmisconception用(unseen)として分けておく
        unseen = df.filter(pl.col("MisconceptionId").is_in(candidate_misconception_ids))

        # unseenを除くデータでfoldを作成
        # 実際にはseenにもunseenが含まれている。これにunseenを結合してunseen_rateを調整する
        seen = df.filter(~pl.col("QuestionId").is_in(unseen["QuestionId"].to_list()))
        seen = get_stratifiedgroupkfold(
            seen, target_col="MisconceptionId", group_col="QuestionId", n_splits=self.cfg.n_splits, seed=self.cfg.seed
        )
        return seen, unseen

    def generate_candidates(self, df: pl.DataFrame, misconception_mapping: pl.DataFrame) -> pl.DataFrame:
        # fine-tuning前のモデルによるembeddingの類似度から負例候補を取得
        model = SentenceTransformer(self.cfg.retrieval_model.name)
        sorted_similarity = sentence_emb_similarity(df, misconception_mapping, model)
        df = df.with_columns(
            pl.Series(sorted_similarity[:, : self.cfg.max_candidates].tolist()).alias("PredictMisconceptionId")
        )
        df = create_retrieved(df, misconception_mapping, self.cfg)
        return df

    def feature_engineering(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cfg.llm.active:
            df = add_llm_misconception_features(df, self.cfg)
        return df

    def run(self) -> None:
        df, misconception = self.read_data()
        df = self.preprocess(df)
        df = self.feature_engineering(df)
        if self.cfg.phase == "train":
            seen, unseen = self.add_fold(df)
            for fold in range(self.cfg.n_splits):
                train, valid = adjust_unseen(seen, unseen, fold, self.cfg.unseen_valid_rate, self.cfg.seed)
                # 必ずtrainとvalidでQuestionIdが被らないようにする
                assert len(set(train["QuestionId"].to_list()) & set(valid["QuestionId"].to_list())) == 0
                train = self.generate_candidates(train, misconception)
                valid = self.generate_candidates(valid, misconception)
                LOGGER.info(f"Train recall: {calc_recall(train):.5f}")
                LOGGER.info(f"Valid recall: {calc_recall(valid):.5f}")
                LOGGER.info(f"Train CV: {calc_mapk(train):.5f}")
                LOGGER.info(f"Valid CV: {calc_mapk(valid):.5f}")
                train.write_csv(self.output_dir / f"train_fold{fold}.csv")
                valid.write_csv(self.output_dir / f"valid_fold{fold}.csv")
                # hold-outで利用するのでfold=0のみ保存
                break
        else:
            df.write_csv(self.output_dir / f"{self.cfg.phase}.csv")


@hydra.main(config_path="./", config_name="config", version_base="1.2")  # type: ignore
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
