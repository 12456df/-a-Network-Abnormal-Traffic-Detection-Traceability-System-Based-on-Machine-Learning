"""
cicflowmeter_preprocessing.py
=============================

使用 CICFlowMeter（Python 版）将 PCAP 文件转换为与 CIC-IDS2017 同结构的 flow 特征 CSV。

依赖（你已安装）：
    pip install cicflowmeter

本脚本通过调用 cicflowmeter 命令行完成转换，支持：
- 单个 pcap 文件 -> 单个 CSV
- 指定目录下所有 pcap -> 每个 pcap 一个 CSV，或合并为一个 CSV

生成后的 CSV 含 80+ 维 flow 特征，可与 MachineLearningCVE 下的数据一起用于训练或对比。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _get_subprocess_env() -> dict[str, str]:
    """
    构造子进程的环境变量，确保 PATH 中包含 tcpdump 所在目录。
    Cursor 等 IDE 的终端可能未继承用户后加的 PATH，导致 cicflowmeter 内 scapy 找不到 tcpdump。
    """
    env = dict(os.environ)
    path_sep = os.pathsep
    tcpdump_path = shutil.which("tcpdump")
    if tcpdump_path:
        tcpdump_dir = os.path.dirname(tcpdump_path)
        env["PATH"] = tcpdump_dir + path_sep + env.get("PATH", "")
    else:
        # 若当前进程找不到 tcpdump，尝试常见路径，便于在 Cursor 终端中也能让子进程找到
        extra_dirs = [r"C:\Tools", r"C:\Tools\WinDump"]
        existing = env.get("PATH", "")
        for d in extra_dirs:
            if os.path.isdir(d) and ((Path(d) / "tcpdump.exe").exists() or (Path(d) / "windump.exe").exists()):
                env["PATH"] = d + path_sep + existing
                break
    return env


def _get_cicflowmeter_cmd() -> list[str]:
    """
    获取用于调用 CICFlowMeter 的命令列表。
    优先使用当前 Python 环境下的 cicflowmeter 可执行文件（Scripts/cicflowmeter），
    避免 python -m cicflowmeter 在部分环境下报「包不能直接执行」。
    """
    exe_dir = Path(sys.executable).resolve().parent
    if sys.platform == "win32":
        script = exe_dir / "Scripts" / "cicflowmeter.exe"
    else:
        script = exe_dir / "Scripts" / "cicflowmeter"
    if script.exists():
        return [str(script)]
    # 若 Scripts 下没有，尝试 PATH 中的 cicflowmeter
    which = shutil.which("cicflowmeter")
    if which:
        return [which]
    # 最后退回 -m，若仍失败请在本环境执行: py -m pip install cicflowmeter
    return [sys.executable, "-m", "cicflowmeter"]


@dataclass
class CICFlowMeterConfig:
    """
    CICFlowMeter 调用相关配置。
    """

    # 输入：单个 pcap 文件（单文件模式时使用）
    pcap_file: Path = PROJECT_ROOT / "rowdata" / "PCAPs" / "Friday-WorkingHours.pcap"

    # 输入：pcap 目录（目录模式时使用，与 pcap_file 二选一）
    pcap_dir: Path | None = None

    # 输出：单文件模式时为 CSV 文件路径；目录模式时为输出目录
    output_path: Path = PROJECT_ROOT / "processed" / "cicflowmeter_friday.csv"

    # 目录模式下是否将所有 pcap 的结果合并为一个 CSV
    merge: bool = False


def _ensure_parent(path: Path) -> None:
    """确保输出路径的父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def run_cicflowmeter_single(pcap_path: Path, csv_path: Path) -> None:
    """
    对单个 pcap 文件运行 CICFlowMeter，输出一个 CSV。
    """
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {pcap_path}")

    _ensure_parent(csv_path)
    cmd = _get_cicflowmeter_cmd() + [
        "-f",
        str(pcap_path),
        "-c",
        str(csv_path),
    ]
    print(f"[cicflowmeter_preprocessing] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=_get_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(f"CICFlowMeter 退出码非 0: {result.returncode}")
    print(f"[cicflowmeter_preprocessing] 已生成: {csv_path}")


def run_cicflowmeter_dir(
    pcap_dir: Path,
    output_dir: Path,
    merge: bool = False,
) -> None:
    """
    对指定目录下所有 .pcap 文件逐个运行 CICFlowMeter（当前安装版仅支持 -f 单文件，不支持 -d 目录）。
    - 每个 pcap 生成一个 CSV 到 output_dir，文件名为 <pcap 原名>.csv。
    - merge=True 时，先逐个生成后再合并为一个 CSV（见下方实现）。
    """
    if not pcap_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {pcap_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pcap_files = sorted(pcap_dir.glob("*.pcap"))
    if not pcap_files:
        print(f"[cicflowmeter_preprocessing] 目录下未找到 .pcap 文件: {pcap_dir}")
        return

    for pcap_path in pcap_files:
        csv_path = output_dir / f"{pcap_path.stem}.csv"
        print(f"[cicflowmeter_preprocessing] 处理: {pcap_path.name} -> {csv_path.name}")
        run_cicflowmeter_single(pcap_path, csv_path)

    if merge and len(pcap_files) > 1:
        # 简单合并：将已生成的 CSV 按顺序拼成一个（需 pandas）
        try:
            import pandas as pd
            dfs = []
            for pcap_path in pcap_files:
                csv_path = output_dir / f"{pcap_path.stem}.csv"
                if csv_path.exists():
                    dfs.append(pd.read_csv(csv_path))
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                merged_path = output_dir / "merged_flows.csv"
                merged.to_csv(merged_path, index=False)
                print(f"[cicflowmeter_preprocessing] 已合并为: {merged_path}")
        except Exception as e:
            print(f"[cicflowmeter_preprocessing] 合并 CSV 时出错（可忽略）: {e}")


def run_cicflowmeter_preprocessing(config: CICFlowMeterConfig | None = None) -> None:
    """
    根据配置执行 CICFlowMeter 转换。
    - 若 config.pcap_dir 不为 None，则按目录模式运行；
    - 否则按单文件模式，使用 config.pcap_file 与 config.output_path。
    """
    if config is None:
        config = CICFlowMeterConfig()

    if config.pcap_dir is not None:
        run_cicflowmeter_dir(
            config.pcap_dir,
            config.output_path,
            merge=config.merge,
        )
    else:
        run_cicflowmeter_single(config.pcap_file, config.output_path)


def main() -> None:
    """
    处理 rowdata/PCAPs/ 下所有 .pcap 文件，每个 pcap 分别生成一个 CSV，保存到 pcap_processed/ 下（不合并）。
    """
    config = CICFlowMeterConfig(
        pcap_dir=PROJECT_ROOT / "rowdata" / "PCAPs",
        output_path=PROJECT_ROOT / "pcap_processed",
        merge=False,
    )
    run_cicflowmeter_preprocessing(config)


if __name__ == "__main__":
    print("[cicflowmeter_preprocessing] 开始使用 CICFlowMeter 处理 PCAP...")
    try:
        main()
    except Exception as exc:
        print(f"[cicflowmeter_preprocessing] 发生错误: {exc}")
        raise
    else:
        print("[cicflowmeter_preprocessing] 处理完成。")
