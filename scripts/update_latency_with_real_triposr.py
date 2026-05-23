import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LOG_FILE = Path("plus_run.log")
DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard")
DEFAULT_TRIPOSR_OUTPUT_DIR = Path("paper_repro_outputs/triposr_latency_analysis")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def run_command(command: list[str]) -> None:
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cache latency reports using real TripoSR timings from logs.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--triposr-output-dir", default=str(DEFAULT_TRIPOSR_OUTPUT_DIR))
    args = parser.parse_args()

    log_file = Path(args.log_file)
    eval_dir = Path(args.eval_dir)
    triposr_output_dir = Path(args.triposr_output_dir)

    if not log_file.exists():
        print(f"未找到日志文件：{log_file}")
        print("请先用如下命令运行 plus.py 并保存日志：")
        print(r".\.venv\Scripts\python.exe plus.py *> plus_run.log")
        return 0

    python_exe = sys.executable
    run_command(
        [
            python_exe,
            "extract_triposr_latency_from_logs.py",
            "--log-file",
            str(log_file),
            "--output-dir",
            str(triposr_output_dir),
        ]
    )

    triposr_summary_path = triposr_output_dir / "triposr_latency_summary.json"
    triposr_summary = read_json(triposr_summary_path)
    avg_success_duration_ms = parse_positive_float(triposr_summary.get("avg_success_duration_ms"))

    if avg_success_duration_ms is None:
        print("未解析到成功的 TripoSR 生成耗时，无法替换 generation_ms。")
        print(f"请检查：{triposr_summary_path}")
        return 0

    run_command(
        [
            python_exe,
            "analyze_cache_latency.py",
            "--eval-dir",
            str(eval_dir),
            "--generation-ms",
            str(avg_success_duration_ms),
        ]
    )

    run_command(
        [
            python_exe,
            "merge_cache_final_with_latency.py",
            "--eval-dir",
            str(eval_dir),
        ]
    )

    latency_report_path = eval_dir / "latency_analysis" / "latency_report.md"
    final_report_path = eval_dir / "cache_teacher_final_report_with_latency.md"

    print("=" * 72)
    print(f"log_file: {log_file}")
    print(f"total_calls: {triposr_summary.get('total_calls')}")
    print(f"success_count: {triposr_summary.get('success_count')}")
    print(f"avg_success_duration_ms: {avg_success_duration_ms}")
    print(f"latency_report.md: {latency_report_path}")
    print(f"cache_teacher_final_report_with_latency.md: {final_report_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
