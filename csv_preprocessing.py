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
from typing import Dict, Iterable, List, Tuple

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
    raw_csv_dir: Path = PROJECT_ROOT / "rowdata" / "CSVs" / "TrafficLabelling"

    # 输出目录（已在仓库中创建 processed/）
    processed_dir: Path = PROJECT_ROOT / "processed"

    # 标签列名称（CIC-IDS2017 中为 "Label"）
    label_col: str = "Label"

    # 二分类标签列名称
    binary_label_col: str = "binary_label"

    # 样本唯一 ID 列名称，用于训练结果回写和溯源关联
    sample_id_col: str = "sample_id"

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
        # TrafficLabelling 含 cp1252 编码字符（如 en-dash），需 fallback
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")
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


def basic_cleaning(
    df: pd.DataFrame, protected_cols: Iterable[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    执行基础数据清洗，返回清洗后的 DataFrame 以及数值特征列名列表。

    清洗内容包括：
    - 对候选特征列进行 to_numeric(errors='coerce')，统一数值化。
    - 将特殊值（inf, -inf）替换为 NaN，并删除候选特征存在 NaN 的样本。
    - 删除常数特征列（所有值相同），但保留受保护列。
    - 返回数值型特征列作为模型输入。
    """
    df = df.copy()

    protected_cols_set = set(protected_cols)
    candidate_feature_cols = [col for col in df.columns if col not in protected_cols_set]

    # 候选特征列统一数值化，无法解析的值会成为 NaN
    for col in candidate_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 仅针对候选特征处理 inf / NaN，避免因为元数据缺失导致样本被误删
    df[candidate_feature_cols] = df[candidate_feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=0, how="any", subset=candidate_feature_cols, inplace=True)

    # 删除常数列（所有值相同），受保护列除外
    constant_cols: List[str] = []
    for col in df.columns:
        if col in protected_cols_set:
            continue
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)

    # 数值特征列 = 所有非受保护列中，dtype 为 number 的列
    numeric_cols = [
        col
        for col in df.columns
        if col not in protected_cols_set and pd.api.types.is_numeric_dtype(df[col])
    ]

    return df, numeric_cols


def split_train_val_test(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    config: PreprocessConfig,
    passthrough_cols: List[str] | None = None,
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

    # 保留样本 ID，便于预测结果回写和攻击溯源关联
    if config.sample_id_col in df.columns:
        train_df[config.sample_id_col] = df.loc[x_train.index, config.sample_id_col].values
        val_df[config.sample_id_col] = df.loc[x_val.index, config.sample_id_col].values
        test_df[config.sample_id_col] = df.loc[x_test.index, config.sample_id_col].values

    # 可选透传额外列（例如同时保留 binary_label 或 Label）
    if passthrough_cols:
        for col in passthrough_cols:
            if col not in df.columns or col == label_col:
                continue
            train_df[col] = df.loc[x_train.index, col].values
            val_df[col] = df.loc[x_val.index, col].values
            test_df[col] = df.loc[x_test.index, col].values

    return train_df, val_df, test_df


def add_sample_id(df: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    """
    为每条样本分配稳定的整型 ID，便于训练、预测与溯源图之间做主键关联。
    """
    df = df.copy()
    df[sample_id_col] = np.arange(len(df), dtype=np.int64)
    return df


def build_trace_outputs(
    full_df: pd.DataFrame,
    binary_train_df: pd.DataFrame,
    binary_val_df: pd.DataFrame,
    binary_test_df: pd.DataFrame,
    multiclass_train_df: pd.DataFrame,
    multiclass_val_df: pd.DataFrame,
    multiclass_test_df: pd.DataFrame,
    config: PreprocessConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    构建溯源相关输出：
    1) trace_metadata：样本级元数据（含 split），可用于将预测结果回写到原始流记录。
    2) trace_graph_edges：图边级聚合结果，可直接用于图论建模/可视化。
    """
    binary_split_map: Dict[int, str] = {}
    if config.sample_id_col in binary_train_df.columns:
        binary_split_map.update(
            {int(v): "train" for v in binary_train_df[config.sample_id_col].values}
        )
    if config.sample_id_col in binary_val_df.columns:
        binary_split_map.update(
            {int(v): "val" for v in binary_val_df[config.sample_id_col].values}
        )
    if config.sample_id_col in binary_test_df.columns:
        binary_split_map.update(
            {int(v): "test" for v in binary_test_df[config.sample_id_col].values}
        )

    multiclass_split_map: Dict[int, str] = {}
    if config.sample_id_col in multiclass_train_df.columns:
        multiclass_split_map.update(
            {int(v): "train" for v in multiclass_train_df[config.sample_id_col].values}
        )
    if config.sample_id_col in multiclass_val_df.columns:
        multiclass_split_map.update(
            {int(v): "val" for v in multiclass_val_df[config.sample_id_col].values}
        )
    if config.sample_id_col in multiclass_test_df.columns:
        multiclass_split_map.update(
            {int(v): "test" for v in multiclass_test_df[config.sample_id_col].values}
        )

    trace_candidate_cols = [
        config.sample_id_col,
        "source_file",
        config.label_col,
        config.binary_label_col,
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Flow Duration",
    ]
    trace_cols = [col for col in trace_candidate_cols if col in full_df.columns]
    trace_metadata = full_df[trace_cols].copy()

    if config.sample_id_col in trace_metadata.columns:
        if binary_split_map:
            trace_metadata["split_binary"] = (
                trace_metadata[config.sample_id_col]
                .map(binary_split_map)
                .fillna("unknown")
            )
        if multiclass_split_map:
            trace_metadata["split_multiclass"] = (
                trace_metadata[config.sample_id_col]
                .map(multiclass_split_map)
                .fillna("unknown")
            )

    # 构建图边数据：
    # 1) 优先使用 Source IP -> Destination IP（标准网络流图）
    # 2) 若缺失，回退为 source_file -> Flow ID
    # 3) 进一步回退为 source_file -> Destination Port
    edge_df = trace_metadata.copy()
    if {"Source IP", "Destination IP"}.issubset(set(edge_df.columns)):
        edge_df["src_node"] = edge_df["Source IP"].astype(str)
        edge_df["dst_node"] = edge_df["Destination IP"].astype(str)
        edge_df["src_node_type"] = "ip"
        edge_df["dst_node_type"] = "ip"
        extra_group_cols = [
            col
            for col in ["Source Port", "Destination Port", "Protocol"]
            if col in edge_df.columns
        ]
    elif {"source_file", "Flow ID"}.issubset(set(edge_df.columns)):
        edge_df["src_node"] = edge_df["source_file"].astype(str)
        edge_df["dst_node"] = edge_df["Flow ID"].astype(str)
        edge_df["src_node_type"] = "file"
        edge_df["dst_node_type"] = "flow"
        extra_group_cols = []
    elif {"source_file", "Destination Port"}.issubset(set(edge_df.columns)):
        edge_df["src_node"] = edge_df["source_file"].astype(str)
        edge_df["dst_node"] = edge_df["Destination Port"].astype(str)
        edge_df["src_node_type"] = "file"
        edge_df["dst_node_type"] = "port"
        extra_group_cols = []
    else:
        return trace_metadata, None

    edge_group_cols = ["src_node", "dst_node", "src_node_type", "dst_node_type", *extra_group_cols]
    if "Timestamp" in edge_df.columns:
        edge_df["Timestamp"] = pd.to_datetime(edge_df["Timestamp"], errors="coerce")

    if config.binary_label_col in edge_df.columns:
        edge_summary = (
            edge_df.groupby(edge_group_cols, dropna=False)
            .agg(
                flow_count=(config.binary_label_col, "size"),
                attack_flow_count=(config.binary_label_col, "sum"),
            )
            .reset_index()
        )
    else:
        edge_summary = (
            edge_df.groupby(edge_group_cols, dropna=False)
            .size()
            .rename("flow_count")
            .reset_index()
        )
        edge_summary["attack_flow_count"] = np.nan

    if "Timestamp" in edge_df.columns:
        ts_summary = (
            edge_df.groupby(edge_group_cols, dropna=False)["Timestamp"]
            .agg(first_seen="min", last_seen="max")
            .reset_index()
        )
        edge_summary = edge_summary.merge(ts_summary, on=edge_group_cols, how="left")

    if "attack_flow_count" in edge_summary.columns:
        edge_summary["attack_flow_ratio"] = np.where(
            edge_summary["flow_count"] > 0,
            edge_summary["attack_flow_count"] / edge_summary["flow_count"],
            0.0,
        )

    return trace_metadata, edge_summary


def ensure_processed_dir(path: Path) -> None:
    """
    确保输出目录存在。
    """
    path.mkdir(parents=True, exist_ok=True)


def save_outputs(
    full_df: pd.DataFrame,
    binary_train_df: pd.DataFrame,
    binary_val_df: pd.DataFrame,
    binary_test_df: pd.DataFrame,
    multiclass_train_df: pd.DataFrame,
    multiclass_val_df: pd.DataFrame,
    multiclass_test_df: pd.DataFrame,
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

    # 二分类训练 / 验证 / 测试集（保留兼容文件名 + 显式命名文件）
    train_path = config.processed_dir / "train.csv"
    val_path = config.processed_dir / "val.csv"
    test_path = config.processed_dir / "test.csv"
    train_binary_path = config.processed_dir / "train_binary.csv"
    val_binary_path = config.processed_dir / "val_binary.csv"
    test_binary_path = config.processed_dir / "test_binary.csv"

    binary_train_df.to_csv(train_path, index=False)
    binary_val_df.to_csv(val_path, index=False)
    binary_test_df.to_csv(test_path, index=False)
    binary_train_df.to_csv(train_binary_path, index=False)
    binary_val_df.to_csv(val_binary_path, index=False)
    binary_test_df.to_csv(test_binary_path, index=False)

    # 多分类训练 / 验证 / 测试集
    train_multiclass_path = config.processed_dir / "train_multiclass.csv"
    val_multiclass_path = config.processed_dir / "val_multiclass.csv"
    test_multiclass_path = config.processed_dir / "test_multiclass.csv"
    multiclass_train_df.to_csv(train_multiclass_path, index=False)
    multiclass_val_df.to_csv(val_multiclass_path, index=False)
    multiclass_test_df.to_csv(test_multiclass_path, index=False)

    # 无监督学习常用输入：全量特征、仅 BENIGN 训练特征
    unsup_full_path = config.processed_dir / "unsupervised_features_full.csv"
    full_df[feature_cols].to_csv(unsup_full_path, index=False)

    unsup_benign_path = config.processed_dir / "unsupervised_benign_train.csv"
    if config.binary_label_col in binary_train_df.columns:
        binary_train_df.loc[binary_train_df[config.binary_label_col] == 0, feature_cols].to_csv(
            unsup_benign_path, index=False
        )
    else:
        binary_train_df[feature_cols].to_csv(unsup_benign_path, index=False)

    # 特征列名
    feature_names_path = config.processed_dir / "feature_names.txt"
    feature_names_path.write_text("\n".join(feature_cols), encoding="utf-8")

    # 标签映射信息（二分类 + 多分类标签空间）
    multiclass_labels = sorted(full_df[config.label_col].astype(str).unique().tolist())
    label_mapping = {
        config.binary_label_col: {
            "0": "BENIGN",
            "1": "ATTACK (all non-BENIGN labels)",
        },
        config.label_col: {str(i): label for i, label in enumerate(multiclass_labels)},
    }
    label_mapping_path = config.processed_dir / "label_mapping.json"
    label_mapping_path.write_text(json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    # 溯源输出：样本级元数据 + 图边级聚合
    trace_metadata, trace_edges = build_trace_outputs(
        full_df=full_df,
        binary_train_df=binary_train_df,
        binary_val_df=binary_val_df,
        binary_test_df=binary_test_df,
        multiclass_train_df=multiclass_train_df,
        multiclass_val_df=multiclass_val_df,
        multiclass_test_df=multiclass_test_df,
        config=config,
    )
    trace_metadata_path = config.processed_dir / "trace_metadata.csv"
    trace_metadata.to_csv(trace_metadata_path, index=False)

    if trace_edges is not None:
        trace_edges_path = config.processed_dir / "trace_graph_edges.csv"
        trace_edges.to_csv(trace_edges_path, index=False)


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

    # TrafficLabelling 版部分文件含无标签行（Label 为 NaN），需先过滤
    n_before = len(combined)
    combined = combined.dropna(subset=[config.label_col])
    combined = combined[combined[config.label_col].astype(str).str.strip() != ""]
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        print(f"[csv_preprocessing] 已过滤 {n_dropped} 行无标签数据")

    # 新增二分类标签列
    combined = add_binary_label(combined, config.label_col, config.binary_label_col)
    combined = add_sample_id(combined, config.sample_id_col)

    # 基础清洗，使用二分类标签作为主要训练目标；并保护溯源字段不被作为特征
    protected_cols = [
        config.label_col,
        config.binary_label_col,
        config.sample_id_col,
        "source_file",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Protocol",
        "Timestamp",
    ]
    cleaned_df, feature_cols = basic_cleaning(combined, protected_cols=protected_cols)
    if not feature_cols:
        raise RuntimeError("清洗后无可用数值特征，请检查输入数据和字段配置。")

    # 输出二分类切分（是否攻击）
    binary_train_df, binary_val_df, binary_test_df = split_train_val_test(
        cleaned_df,
        label_col=config.binary_label_col,
        feature_cols=feature_cols,
        config=config,
        passthrough_cols=[config.label_col],
    )

    # 输出多分类切分（攻击类别识别）
    multiclass_train_df, multiclass_val_df, multiclass_test_df = split_train_val_test(
        cleaned_df,
        label_col=config.label_col,
        feature_cols=feature_cols,
        config=config,
        passthrough_cols=[config.binary_label_col],
    )

    save_outputs(
        full_df=cleaned_df,
        binary_train_df=binary_train_df,
        binary_val_df=binary_val_df,
        binary_test_df=binary_test_df,
        multiclass_train_df=multiclass_train_df,
        multiclass_val_df=multiclass_val_df,
        multiclass_test_df=multiclass_test_df,
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

