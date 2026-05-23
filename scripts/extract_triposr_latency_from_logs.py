import argparse
import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_LOG_CANDIDATES = [
    Path("paper_repro_outputs/logs/plus_run.log"),
    Path("runtime_assets/plus_run.log"),
    Path("plus_run.log"),
]
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/triposr_latency_analysis")
TIMESTAMP_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")


@dataclass
class TripoSRRecord:
    index: int
    start_time: str
    end_time: str
    duration_ms: float
    duration_seconds: float
    status: str
    start_line: int
    end_line: int


def parse_timestamp(line: str) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def find_log_file(user_log_file: Optional[str]) -> Optional[Path]:
    if user_log_file:
        path = Path(user_log_file)
        return path if path.exists() else None
    for candidate in DEFAULT_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def classify_end_line(line: str) -> Optional[str]:
    if "TripoSR 模型已缓存" in line:
        return "success"
    if "TripoSR returncode" in line or "TripoSR 调用异常" in line:
        return "failed"
    return None


def parse_records(log_file: Path) -> List[TripoSRRecord]:
    records: List[TripoSRRecord] = []
    active_start_time: Optional[datetime] = None
    active_start_line: Optional[int] = None

    with log_file.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            timestamp = parse_timestamp(line)
            if timestamp is None:
                continue

            if "调用 TripoSR" in line:
                if active_start_time is not None and active_start_line is not None:
                    duration_ms = (timestamp - active_start_time).total_seconds() * 1000
                    records.append(
                        TripoSRRecord(
                            index=len(records) + 1,
                            start_time=active_start_time.isoformat(sep=" "),
                            end_time=timestamp.isoformat(sep=" "),
                            duration_ms=round(duration_ms, 3),
                            duration_seconds=round(duration_ms / 1000.0, 3),
                            status="unknown",
                            start_line=active_start_line,
                            end_line=line_no,
                        )
                    )
                active_start_time = timestamp
                active_start_line = line_no
                continue

            status = classify_end_line(line)
            if status and active_start_time is not None and active_start_line is not None:
                duration_ms = (timestamp - active_start_time).total_seconds() * 1000
                records.append(
                    TripoSRRecord(
                        index=len(records) + 1,
                        start_time=active_start_time.isoformat(sep=" "),
                        end_time=timestamp.isoformat(sep=" "),
                        duration_ms=round(duration_ms, 3),
                        duration_seconds=round(duration_ms / 1000.0, 3),
                        status=status,
                        start_line=active_start_line,
                        end_line=line_no,
                    )
                )
                active_start_time = None
                active_start_line = None

    if active_start_time is not None and active_start_line is not None:
        records.append(
            TripoSRRecord(
                index=len(records) + 1,
                start_time=active_start_time.isoformat(sep=" "),
                end_time="",
                duration_ms=0.0,
                duration_seconds=0.0,
                status="unknown",
                start_line=active_start_line,
                end_line=0,
            )
        )

    return records


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.median(values), 3)


def make_summary(records: List[TripoSRRecord]) -> Dict[str, Any]:
    completed = [record.duration_ms for record in records if record.duration_ms > 0]
    success = [record.duration_ms for record in records if record.status == "success" and record.duration_ms > 0]
    failed = [record.duration_ms for record in records if record.status == "failed" and record.duration_ms > 0]

    return {
        "total_calls": len(records),
        "success_count": sum(1 for record in records if record.status == "success"),
        "failed_count": sum(1 for record in records if record.status == "failed"),
        "unknown_count": sum(1 for record in records if record.status == "unknown"),
        "avg_duration_ms": average(completed),
        "median_duration_ms": median(completed),
        "min_duration_ms": round(min(completed), 3) if completed else None,
        "max_duration_ms": round(max(completed), 3) if completed else None,
        "avg_success_duration_ms": average(success),
        "avg_failed_duration_ms": average(failed),
    }


def write_records_csv(path: Path, records: List[TripoSRRecord]) -> None:
    fieldnames = [
        "index",
        "start_time",
        "end_time",
        "duration_ms",
        "duration_seconds",
        "status",
        "start_line",
        "end_line",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_report(path: Path, log_file: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# TripoSR 真实生成耗时统计",
        "",
        "## 1. 统计目的",
        "",
        "本统计用于从已有 plus.py 运行日志中提取 TripoSR 调用耗时，替换缓存延迟分析中的估计 generation_ms，使导师汇报中的等待时间对比更严谨。",
        "",
        "## 2. 数据来源",
        "",
        f"- log_file: {log_file}",
        "",
        "## 3. 统计结果",
        "",
        f"- total_calls: {summary.get('total_calls')}",
        f"- success_count: {summary.get('success_count')}",
        f"- failed_count: {summary.get('failed_count')}",
        f"- avg_duration_ms: {summary.get('avg_duration_ms')}",
        f"- median_duration_ms: {summary.get('median_duration_ms')}",
        f"- min_duration_ms: {summary.get('min_duration_ms')}",
        f"- max_duration_ms: {summary.get('max_duration_ms')}",
        f"- avg_success_duration_ms: {summary.get('avg_success_duration_ms')}",
        "",
        "## 4. 后续使用",
        "",
        "后续可将 avg_success_duration_ms 作为 analyze_cache_latency.py 的 --generation-ms 参数重新生成延迟报告。例如：",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe analyze_cache_latency.py --generation-ms <avg_success_duration_ms>",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract TripoSR generation latency from plus.py logs.")
    parser.add_argument("--log-file")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    log_file = find_log_file(args.log_file)
    if log_file is None:
        print("未找到日志文件，请将 plus.py 运行日志保存为 plus_run.log 后重试。")
        if args.log_file:
            print(f"指定路径不存在：{args.log_file}")
        else:
            print("已尝试默认路径：")
            for candidate in DEFAULT_LOG_CANDIDATES:
                print(f"  - {candidate}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_records(log_file)
    summary = make_summary(records)
    summary["log_file"] = str(log_file)

    records_path = output_dir / "triposr_latency_records.csv"
    summary_path = output_dir / "triposr_latency_summary.json"
    report_path = output_dir / "triposr_latency_report.md"

    write_records_csv(records_path, records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, log_file, summary)

    print("=" * 72)
    print(f"log_file: {log_file}")
    print(f"total_calls: {summary.get('total_calls')}")
    print(f"success_count: {summary.get('success_count')}")
    print(f"failed_count: {summary.get('failed_count')}")
    print(f"avg_success_duration_ms: {summary.get('avg_success_duration_ms')}")
    print(f"triposr_latency_report.md: {report_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
