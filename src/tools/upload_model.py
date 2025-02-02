import json
import shutil
from typing import Any
from pathlib import Path

import click
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_TITLE = "kuto-eedi-model"  # ここをコンペごとに変更

TARGET_EXP_RUN_NAMES = [
    # "feature_store",
    "dummy/run0",
    # "exp001",
    # "exp002",
    # "exp003",
    # "exp005",
    # "exp006",
    # "exp008",
    # "exp009",
    # "exp010",
    # "exp011",
    # "exp013",
    # "exp016",
    # "exp021",
    # "exp023",
    # "exp025",
    # "exp026",
    # "exp029",
    # "exp030",
    # "exp037/run9_dunzhang-stella_en_1.5B_v5_epoch20_top50_alpha512",
    # "exp037/run3_dunzhang-stella_en_1.5B_v5_epoch20_top100",
    # "exp040/run13_dunzhang-stella_en_1.5B_v5_multinega10_epoch10_candidate50_lora_alpha512_lr8e-06",
    # "exp040/run13",
    # "exp042/run0",
    # "exp042/run1",
    # "exp042/run2",
    # "exp042/run3",
    # "exp046/run0",
    # "exp046/run1",
    # "exp046/run2",
    # "exp046/run3",
    # "exp046/run4",
    # "exp048/run0",
    # "exp049/run0",
    # "exp049/run1",
    # "exp049/run2",
    # "exp049/run3",
    # "exp049/run4",
    # "exp050/run6",
    # "exp050/run18",
    # "exp050/run23",
    # "exp050/run0",
    # "exp052/run3",
    # "exp052/run5",
    # "exp052/run6",
    # "exp052/run8",
    "exp053/run0",
    # "exp054/run1",
    # "exp054/run2",
    # "exp054/run3",
    # "exp054/run4",
    "exp055/run2",
]


def copy_files_with_exts(source_dir: Path, dest_dir: Path, exts: list) -> None:
    """
    source_dir: 探索開始ディレクトリ
    dest_dir: コピー先のディレクトリ
    exts: 対象の拡張子のリスト (例: ['.txt', '.jpg'])
    """

    # source_dirの中での各拡張子と一致するファイルのパスを探索
    for ext in exts:
        for source_path in source_dir.rglob(f"*{ext}"):
            # dest_dir内での相対パスを計算
            relative_path = source_path.relative_to(source_dir)
            exp_run_name = "/".join(str(relative_path).split("/")[:-1])

            if exp_run_name not in TARGET_EXP_RUN_NAMES:
                continue

            dest_path = dest_dir / relative_path

            # 必要に応じてコピー先ディレクトリを作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # ファイルをコピー
            shutil.copy2(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")


@click.command()
@click.option("--title", "-t", default=DATASET_TITLE)
@click.option("--dir", "-d", type=Path, default="./output/")
@click.option(
    "--extentions",
    "-e",
    type=list[str],
    default=[
        "model.safetensors",
        "vocab.txt",
        ".json",
        ".py",
        # ".hydra/*.yaml",
        # "preds.npy",
    ],
)
@click.option("--user_name", "-u", default="kuto0633")
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    dir: Path,
    extentions: list[str],
    user_name: str,
    new: bool = False,
) -> None:
    """extentionを指定して、dir以下のファイルをzipに圧縮し、kaggleにアップロードする。

    Args:
        title (str): kaggleにアップロードするときのタイトル
        dir (Path): アップロードするファイルがあるディレクトリ
        extentions (list[str], optional): アップロードするファイルの拡張子.
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
    """
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 拡張子が.pthのファイルをコピー
    copy_files_with_exts(dir, tmp_dir, extentions)

    # dataset-metadata.jsonを作成
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # api認証
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )

    # delete tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
