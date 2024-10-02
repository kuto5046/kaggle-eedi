from pathlib import Path

import polars as pl
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

INPUT_DIR = Path("/home/user/work/input/eedi-mining-misconceptions-in-mathematics")


def replace_latex(text: str) -> str:
    return text.replace(r"\[", "$").replace(r"\]", "$").replace(r"\(", "$").replace(r"\)", "$")


def plot_answer(
    col: DeltaGenerator,
    answer: str,
    misconception_id: float | None,
    alphabet: str,
    is_correct_answer: bool,
    misconception_mapping: dict[int, str],
) -> None:
    if misconception_id is not None:
        misconception_id = int(misconception_id)
        misconception_name = misconception_mapping[misconception_id]
    else:
        misconception_name = ""

    with col:
        if is_correct_answer:
            st.write(f"{alphabet} (Correct Answer)")
            st.success(f"Answer: {replace_latex(answer)}")
            st.success(f"Misconception: {misconception_name} ({misconception_id})")
        else:
            st.write(alphabet)
            st.error(f"Answer: {replace_latex(answer)}")
            st.error(f"Misconception: {misconception_name} ({misconception_id})")


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Eedi visualizer")

    df = pl.read_csv(INPUT_DIR / "train.csv")
    misconception_mapping = pl.read_csv(INPUT_DIR / "misconception_mapping.csv")
    misconception_mapping = dict(
        zip(misconception_mapping["MisconceptionId"].to_list(), misconception_mapping["MisconceptionName"].to_list())
    )
    with st.expander("Misconception Mapping"):
        st.write(misconception_mapping)

    with st.sidebar:
        st.markdown("## Filter")
        # id = st.selectbox(label="QuestionId", options=df["QuestionId"].to_list())
        id = st.number_input(label="QuestionId", min_value=0, max_value=df["QuestionId"].max())
    _data = df[id].to_dict(as_series=False)
    data = {k: v[0] for k, v in _data.items()}

    st.info(f"ConstructName: {data['ConstructName']}")
    st.info(f"SubjectName: {data['SubjectName']}")
    st.info(f"QuestionText: {replace_latex(data['QuestionText'])}")

    col1, col2, col3, col4 = st.columns(4)
    plot_answer(
        col1, data["AnswerAText"], data["MisconceptionAId"], "A", data["CorrectAnswer"] == "A", misconception_mapping
    )
    plot_answer(
        col2, data["AnswerBText"], data["MisconceptionBId"], "B", data["CorrectAnswer"] == "B", misconception_mapping
    )
    plot_answer(
        col3, data["AnswerCText"], data["MisconceptionCId"], "C", data["CorrectAnswer"] == "C", misconception_mapping
    )
    plot_answer(
        col4, data["AnswerDText"], data["MisconceptionDId"], "D", data["CorrectAnswer"] == "D", misconception_mapping
    )


if __name__ == "__main__":
    main()
