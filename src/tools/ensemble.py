import pandas as pd
import polars as pl


# https://www.kaggle.com/code/titericz/h-m-ensembling-how-to
def blend(dt: pd.Series, weights: list[float] | None, k: int = 25) -> str:
    # Create a list of all model predictions
    pred_cols = [col for col in list(dt.index) if "pred" in col]
    if weights is None:
        weights = [1.0] * len(pred_cols)
    preds = [dt[col].split() for col in pred_cols]
    # Create a dictionary of items recommended.
    # Assign a weight according the order of appearance and multiply by global weights
    scores: dict[str, float] = {}
    for i in range(len(preds)):
        w = weights[i]
        for rank, pred_misconception_id in enumerate(preds[i]):
            if pred_misconception_id in scores:
                scores[pred_misconception_id] += w / (rank + 1)
            else:
                scores[pred_misconception_id] = w / (rank + 1)

    # Sort dictionary by item weights
    result = list(dict(sorted(scores.items(), key=lambda item: -item[1])).keys())
    return " ".join(result[:k])


# subファイルのlistを入力としてアンサンブルを行う
def ensemble_predictions(subs: list[pl.DataFrame], weights: list[float] | None = None) -> pl.DataFrame:
    if weights is None:
        weights = [1] * len(subs)
    assert len(subs) == len(weights)
    assert len(subs) > 1

    retrieve_num = len(subs[0]["MisconceptionId"].head(1).to_list()[0].split(" "))
    sub = subs[0].select("QuestionId_Answer").clone()
    sub = sub.with_columns([pl.lit(sub["MisconceptionId"]).alias(f"pred{i}") for i, sub in enumerate(subs)]).to_pandas()
    sub["MisconceptionId"] = sub.apply(blend, weights=weights, k=retrieve_num, axis=1)
    return pl.DataFrame(sub[["QuestionId_Answer", "MisconceptionId"]])
