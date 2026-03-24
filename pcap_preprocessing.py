"""
pcap_preprocessing.py
======================

示范性地从 PCAP 文件中提取 flow 级特征。

设计目标（对应你的需求分析）：
- 能够从原始 PCAP 中解析出若干 flow（5 元组 + 时间窗口）。
- 为每条 flow 计算基础统计特征（包数、字节数、持续时间、平均包长等）。
- 将结果保存为 CSV，便于后续检测模型或溯源模块使用。

注意：
- 这是一个“小规模原型”，建议仅对单个 PCAP 的部分时间/部分包做解析，
  避免一次性加载整天的流量导致内存占用过大。
- 依赖第三方库 pyshark（基于 tshark），需要你在本地先安装：
    pip install pyshark
  并确保系统已安装 Wireshark/tshark。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PcapConfig:
    """
    PCAP 解析相关配置。

    你可以按需要修改默认值，或者在 main() 中传入自定义配置。
    """

    # PCAP 文件路径（默认使用 Friday-WorkingHours.pcap）
    pcap_path: Path = PROJECT_ROOT / "rowdata" / "PCAPs" / "Friday-WorkingHours.pcap"

    # 输出 CSV 路径
    output_csv: Path = PROJECT_ROOT / "processed" / "pcap_demo_flows.csv"

    # flow 超时时间（秒）：超过该间隔则认为是新的 flow
    flow_timeout: float = 60.0

    # 最多解析的数据包数量（防止一次加载过多，None 表示不限制）
    max_packets: int | None = 100_000


def _import_pyshark():
    """
    延迟导入 pyshark，避免在未使用本模块时强制要求依赖。
    """
    try:
        import pyshark  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖缺失时提示用户
        raise ImportError(
            "缺少依赖 pyshark，请先在当前 Python 环境中安装：\n"
            "    pip install pyshark\n"
            "并确保系统已安装 Wireshark/tshark。"
        ) from exc
    return pyshark


def _ensure_parent_dir(path: Path) -> None:
    """
    确保输出文件的父目录存在。
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_pcap_to_flows(config: PcapConfig) -> pd.DataFrame:
    """
    将指定 PCAP 文件解析为若干 flow，并返回包含统计特征的 DataFrame。

    解析策略（简化版）：
    - 使用 (src_ip, dst_ip, src_port, dst_port, protocol) 作为 flow key。
    - 以数据包时间戳为基础，若新包距离当前 flow 的最后一个包超过 flow_timeout，
      则开启新的 flow。
    - 对每个 flow 统计包数、总字节数、持续时间、平均/最小/最大包长等指标。
    """
    pyshark = _import_pyshark()

    if not config.pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {config.pcap_path}")

    print(f"[pcap_preprocessing] 开始解析 PCAP: {config.pcap_path}")

    capture = pyshark.FileCapture(
        str(config.pcap_path),
        keep_packets=False,  # 不将所有包都保存在内存
    )

    flows: Dict[Tuple[str, str, str, str, str, int], Dict[str, object]] = {}

    def _get_float_time(pkt) -> float:
        # pyshark 的 sniff_timestamp 是字符串形式的秒数
        return float(pkt.sniff_timestamp)

    def _get_len(pkt) -> int:
        try:
            return int(pkt.length)
        except Exception:
            # length 字段缺失时退化为 0
            return 0

    packet_count = 0

    for pkt in capture:
        packet_count += 1
        if config.max_packets is not None and packet_count > config.max_packets:
            print(
                f"[pcap_preprocessing] 已达到最大解析包数 {config.max_packets}，"
                "提前结束解析。"
            )
            break

        try:
            # 仅处理 TCP、UDP、ICMP 包，其余协议暂不考虑
            if hasattr(pkt, "ip"):
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst
            else:
                # 非 IP 包直接跳过
                continue

            if hasattr(pkt, "tcp"):
                proto = "TCP"
                src_port = pkt.tcp.srcport
                dst_port = pkt.tcp.dstport
            elif hasattr(pkt, "udp"):
                proto = "UDP"
                src_port = pkt.udp.srcport
                dst_port = pkt.udp.dstport
            elif hasattr(pkt, "icmp"):
                proto = "ICMP"
                src_port = "0"
                dst_port = "0"
            else:
                # 其他协议暂不处理
                continue

            ts = _get_float_time(pkt)
            length = _get_len(pkt)
        except Exception:
            # 对于解析异常的包，直接跳过
            continue

        # 使用简单的 key：5 元组 + “第几段 flow” 序号
        base_key = (src_ip, dst_ip, src_port, dst_port, proto)

        # 查找当前已有的 flow（最后一个）
        # 这里的实现是：以递增序号扩展 key，直到找到合适的 flow 或创建新 flow
        seg_index = 0
        while True:
            flow_key = (*base_key, seg_index)
            flow = flows.get(flow_key)
            if flow is None:
                # 新 flow
                flows[flow_key] = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": int(src_port),
                    "dst_port": int(dst_port),
                    "protocol": proto,
                    "start_time": ts,
                    "end_time": ts,
                    "packet_count": 1,
                    "byte_count": length,
                    "min_pkt_len": length,
                    "max_pkt_len": length,
                    "pcap_file": config.pcap_path.name,
                }
                # 简单的进度提示：每创建一条新的 flow 打印一次（可按需注释）
                print(
                    "[pcap_preprocessing] 新建 flow: "
                    f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} ({proto}), "
                    f"start_ts={ts}"
                )
                break

            # 判断是否还能归入当前 flow（未超时）
            last_ts = flow["end_time"]  # type: ignore[assignment]
            if ts - float(last_ts) <= config.flow_timeout:
                # 更新统计数据
                flow["end_time"] = ts  # type: ignore[assignment]
                flow["packet_count"] = int(flow["packet_count"]) + 1  # type: ignore[index]
                flow["byte_count"] = int(flow["byte_count"]) + length  # type: ignore[index]
                flow["min_pkt_len"] = min(int(flow["min_pkt_len"]), length)  # type: ignore[index]
                flow["max_pkt_len"] = max(int(flow["max_pkt_len"]), length)  # type: ignore[index]
                break

            # 若超时，则尝试下一个 seg_index，创建新 flow
            seg_index += 1

    # 转为 DataFrame 并计算衍生特征
    records: List[Dict[str, object]] = []
    for idx, (_, flow) in enumerate(flows.items()):
        start_time = float(flow["start_time"])  # type: ignore[assignment]
        end_time = float(flow["end_time"])  # type: ignore[assignment]
        duration = max(end_time - start_time, 0.0)
        pkt_cnt = int(flow["packet_count"])  # type: ignore[index]
        byte_cnt = int(flow["byte_count"])  # type: ignore[index]
        min_len = int(flow["min_pkt_len"])  # type: ignore[index]
        max_len = int(flow["max_pkt_len"])  # type: ignore[index]
        avg_len = byte_cnt / pkt_cnt if pkt_cnt > 0 else 0.0
        pps = pkt_cnt / duration if duration > 0 else float(pkt_cnt)

        record = {
            "flow_id": idx,
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "packet_count": pkt_cnt,
            "byte_count": byte_cnt,
            "min_pkt_len": min_len,
            "max_pkt_len": max_len,
            "avg_pkt_len": avg_len,
            "pps": pps,
            "pcap_file": flow["pcap_file"],
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    print(
        f"[pcap_preprocessing] 解析完成，共得到 {len(df)} 条 flow "
        f"(来自 {packet_count} 个数据包)。"
    )
    return df


def save_flows_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    将 flow 特征保存为 CSV 文件。
    """
    _ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"[pcap_preprocessing] 已保存 flow 特征到: {output_path}")


def run_pcap_preprocessing(config: PcapConfig | None = None) -> None:
    """
    执行从 PCAP 到 flow CSV 的完整流程。
    """
    if config is None:
        config = PcapConfig()

    df_flows = parse_pcap_to_flows(config)
    save_flows_to_csv(df_flows, config.output_csv)


def main() -> None:
    """
    脚本入口。

    在命令行中执行：
        python pcap_preprocessing.py

    即可对默认的 Friday-WorkingHours.pcap 解析并在 processed/ 下生成
    pcap_demo_flows.csv 示例文件。
    """
    run_pcap_preprocessing()


if __name__ == "__main__":
    print("[pcap_preprocessing] 开始执行 PCAP 预处理...")
    try:
        main()
    except Exception as exc:  # pragma: no cover - 运行时异常直接打印
        print(f"[pcap_preprocessing] 发生错误: {exc}")
        raise
    else:
        print("[pcap_preprocessing] 预处理完成。")

