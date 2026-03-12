"""
csv_preprocessing.py
=====================

面向 CIC-IDS2017（MachineLearningCVE 目录）等公开数据集的 CSV 预处理脚本。

主要职责：
1. 发现并读取原始 CSV 文件（默认：rowdata/CSVs/MachineLearningCVE 下的所有 .csv）。
2. 合并为一张大表，并进行基础数据清洗。
3. 将原始标签列 Label 转换为二分类标签：0 = BENIGN, 1 = ATTACK(所有非 BENIGN)。
4. 按比例划分训练 / 验证 / 测试集。
5. 将结果写入项目根目录下的 processed/ 目录，供模型模块直接使用。

注意：
- 本文件只做数据预处理，不涉及模型训练。
- 若后续需要支持多分类或其他数据集，可以在不破坏现有接口的前提下扩展。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PreprocessConfig:
    """
    预处理过程中的关键配置。

    如果后续需要支持更多数据集或调整划分比例，只需在此处修改默认值，
    或在 main() 中构造自定义配置传入 run_preprocessing。
    """

    # 原始 CSV 所在目录（默认使用 CIC-IDS2017 MachineLearningCVE）
    raw_csv_dir: Path = PROJECT_ROOT / "rowdata" / "CSVs" / "MachineLearningCVE"

    # 输出目录（已在仓库中创建 processed/）
    processed_dir: Path = PROJECT_ROOT / "processed"

    # 标签列名称（CIC-IDS2017 中为 "Label"）
    label_col: str = "Label"

    # 二分类标签列名称
    binary_label_col: str = "binary_label"

    # 训练 / 验证 / 测试集划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # 剩余部分中再划分验证/测试

    # 随机种子，保证实验可复现
    random_state: int = 42


def find_raw_csv_files(config: PreprocessConfig) -> List[Path]:
    """
    搜索原始 CSV 文件列表。
    默认查找 config.raw_csv_dir 下的所有 .csv 文件。
    """
    if not config.raw_csv_dir.exists():
        raise FileNotFoundError(f"原始 CSV 目录不存在: {config.raw_csv_dir}")

    csv_files = sorted(config.raw_csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在目录中未找到任何 CSV 文件: {config.raw_csv_dir}")
    return csv_files


def load_and_concat(csv_files: Iterable[Path], label_col: str) -> pd.DataFrame:
    """
    读取多个 CSV 并纵向合并。

    为了方便后续分析，增加一列 `source_file` 记录每一行来自哪个 CSV。
    若某个文件缺少标签列，则跳过并给出提示，而不是直接报错中断。
    """
    dataframes: List[pd.DataFrame] = []
    for path in csv_files:
        # 读取时不依赖首列前空格，读取后统一去掉列名两端的空白
        df = pd.read_csv(path)
        df.columns = df.columns.astype(str).str.strip()
        # 标记来源文件
        df["source_file"] = path.name

        if label_col not in df.columns:
            # 跳过无标签列的文件，给出清晰提示
            print(
                f"[csv_preprocessing] 跳过文件（缺少标签列 {label_col}）: {path}"
            )
            continue

        dataframes.append(df)

    if not dataframes:
        raise RuntimeError(
            f"在提供的 CSV 文件中，没有任何一个包含标签列 {label_col}，"
            f"请检查数据集或配置。"
        )

    combined = pd.concat(dataframes, ignore_index=True)
    return combined


def add_binary_label(df: pd.DataFrame, label_col: str, binary_label_col: str) -> pd.DataFrame:
    """
    根据原始标签列新增二分类标签列：
    - BENIGN -> 0
    - 其他任意标签 -> 1
    """
    if label_col not in df.columns:
        raise KeyError(f"数据中不存在标签列 {label_col}")

    # 统一处理标签字符串，避免大小写或空白差异
    labels = df[label_col].astype(str).str.strip()
    binary = np.where(labels == "BENIGN", 0, 1)
    df[binary_label_col] = binary
    return df


def basic_cleaning(df: pd.DataFrame, label_cols: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    执行基础数据清洗，返回清洗后的 DataFrame 以及数值特征列名列表。

    清洗内容包括：
    - 将特殊值（inf, -inf）替换为 NaN，并删除存在 NaN 的样本（简单策略，后续可改进）。
    - 删除常数列（所有值相同的列），但保留标签列。
    - 仅保留数值型特征列作为模型输入（标签列除外）。
    """
    df = df.copy()

    # 替换 inf / -inf 为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 简单策略：丢弃包含 NaN 的行
    df.dropna(axis=0, how="any", inplace=True)

    # 删除常数列（所有值相同），标签列除外
    label_cols_set = set(label_cols)
    constant_cols: List[str] = []
    for col in df.columns:
        if col in label_cols_set:
            continue
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)

    # 数值特征列 = 所有非标签列中，dtype 为 number 的列
    numeric_cols = [
        col
        for col in df.columns
        if col not in label_cols_set and pd.api.types.is_numeric_dtype(df[col])
    ]

    return df, numeric_cols


def split_train_val_test(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    config: PreprocessConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按照配置比例，将数据划分为 train / val / test。
    使用 stratify 保持各子集中标签比例一致。
    """
    features = df[feature_cols]
    labels = df[label_col]

    # 先划分 train 和剩余 (val+test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        train_size=config.train_ratio,
        random_state=config.random_state,
        stratify=labels,
    )

    # 再从剩余中划分 val 和 test
    val_ratio_adjusted = config.val_ratio / (1.0 - config.train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        train_size=val_ratio_adjusted,
        random_state=config.random_state,
        stratify=y_temp,
    )

    train_df = x_train.copy()
    train_df[label_col] = y_train.values

    val_df = x_val.copy()
    val_df[label_col] = y_val.values

    test_df = x_test.copy()
    test_df[label_col] = y_test.values

    return train_df, val_df, test_df


def ensure_processed_dir(path: Path) -> None:
    """
    确保输出目录存在。
    """
    path.mkdir(parents=True, exist_ok=True)


def save_outputs(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    config: PreprocessConfig,
) -> None:
    """
    将预处理后的结果写入 processed/ 目录。
    """
    ensure_processed_dir(config.processed_dir)

    # 完整清洗后的数据
    full_path = config.processed_dir / "clean_full.csv"
    full_df.to_csv(full_path, index=False)

    # 训练 / 验证 / 测试集
    train_path = config.processed_dir / "train.csv"
    val_path = config.processed_dir / "val.csv"
    test_path = config.processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 特征列名
    feature_names_path = config.processed_dir / "feature_names.txt"
    feature_names_path.write_text("\n".join(feature_cols), encoding="utf-8")

    # 标签映射信息（当前为二分类）
    label_mapping = {
        config.binary_label_col: {
            "0": "BENIGN",
            "1": "ATTACK (all non-BENIGN labels)",
        }
    }
    label_mapping_path = config.processed_dir / "label_mapping.json"
    label_mapping_path.write_text(json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def run_preprocessing(config: PreprocessConfig | None = None) -> None:
    """
    执行完整的 CSV 预处理流程。

    典型使用方式：
    - 作为脚本直接运行（参见 main）
    - 或在其他模块中导入并调用 run_preprocessing()
    """
    if config is None:
        config = PreprocessConfig()

    csv_files = find_raw_csv_files(config)
    combined = load_and_concat(csv_files, config.label_col)

    # 新增二分类标签列
    combined = add_binary_label(combined, config.label_col, config.binary_label_col)

    # 基础清洗，使用二分类标签作为主要训练目标
    label_cols = [config.label_col, config.binary_label_col]
    cleaned_df, feature_cols = basic_cleaning(combined, label_cols=label_cols)

    # 为后续训练方便，这里默认使用二分类标签列作为目标
    train_df, val_df, test_df = split_train_val_test(
        cleaned_df,
        label_col=config.binary_label_col,
        feature_cols=feature_cols,
        config=config,
    )

    save_outputs(
        full_df=cleaned_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        config=config,
    )


def main() -> None:
    """
    脚本入口。

    在命令行中执行：
        python csv_preprocessing.py

    即可完成一次从原始 CSV 到 processed/ 下各结果文件的预处理流程。
    """
    run_preprocessing()


if __name__ == "__main__":
    # 增加简单的命令行提示，便于在终端中观察执行情况
    print("[csv_preprocessing] 开始预处理 CSV 数据...")
    try:
        main()
    except Exception as exc:
        # 将异常打印出来，方便排查问题，然后继续抛出以便看到完整堆栈
        print(f"[csv_preprocessing] 发生错误: {exc}")
        raise
    else:
        print("[csv_preprocessing] 预处理完成，请查看 processed/ 目录中的输出文件。")

