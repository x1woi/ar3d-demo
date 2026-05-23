import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")


def run_command(args):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
        }

    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }


def detect_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {
            "available": False,
            "path": None,
            "gpus": [],
            "error": "nvidia-smi not found",
        }

    query = run_command(
        [
            exe,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query["ok"] and query["stdout"]:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    memory_mb = int(float(parts[1]))
                except ValueError:
                    memory_mb = None
                gpus.append(
                    {
                        "name": name,
                        "vram_mb": memory_mb,
                        "vram_gb": round(memory_mb / 1024, 2) if memory_mb else None,
                    }
                )

    return {
        "available": True,
        "path": exe,
        "gpus": gpus,
        "error": "" if query["ok"] else query["stderr"] or query["stdout"],
    }


def detect_torch():
    info = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "error": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["available"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if info["cuda_available"]:
        try:
            info["cuda_device_count"] = torch.cuda.device_count()
            for idx in range(info["cuda_device_count"]):
                props = torch.cuda.get_device_properties(idx)
                total_mb = int(props.total_memory / (1024 * 1024))
                info["cuda_devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "vram_mb": total_mb,
                        "vram_gb": round(total_mb / 1024, 2),
                    }
                )
        except Exception as exc:
            info["error"] = str(exc)
    return info


def detect_tool(name, version_args=None):
    exe = shutil.which(name)
    if not exe:
        return {"available": False, "path": None, "version": None, "error": f"{name} not found"}

    args = [exe] + (version_args or ["--version"])
    result = run_command(args)
    version_text = ""
    if result["stdout"]:
        version_text = result["stdout"].splitlines()[0]
    elif result["stderr"]:
        version_text = result["stderr"].splitlines()[0]
    return {
        "available": True,
        "path": exe,
        "version": version_text,
        "error": "" if result["ok"] else result["stderr"],
    }


def get_disk_info(path):
    resolved = Path(path).resolve()
    usage = shutil.disk_usage(str(resolved.anchor or resolved))
    return {
        "path": str(resolved),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def choose_gpu_source(nvidia_info, torch_info):
    if nvidia_info.get("gpus"):
        return nvidia_info["gpus"]
    if torch_info.get("cuda_devices"):
        return torch_info["cuda_devices"]
    return []


def make_recommendation(gpus, torch_info, disk_info):
    max_vram = max((gpu.get("vram_gb") or 0 for gpu in gpus), default=0)
    cuda_available = bool(torch_info.get("cuda_available")) or bool(gpus)
    free_gb = disk_info.get("free_gb") or 0
    warnings = []

    if not cuda_available:
        recommended_backend = "cloud_or_keep_triposr_baseline"
        route = "路线 C：当前未检测到可用 CUDA/GPU，不建议本机尝试高质量 3D 后端，建议云端试跑或继续保留 TripoSR baseline。"
    elif max_vram >= 24:
        recommended_backend = "Hunyuan3D-2.1"
        route = "路线 A：显存较充足，可优先小样本尝试 Hunyuan3D-2.1，同时保留 TripoSR 作为 baseline/fallback。"
    elif 8 <= max_vram <= 12:
        recommended_backend = "Stable Fast 3D or Hunyuan3D-2.0 Turbo/mini"
        route = "路线 B：显存处于 8GB～12GB，更适合先试 Stable Fast 3D，或尝试 Hunyuan3D-2.0 Turbo/mini。"
    elif max_vram > 12:
        recommended_backend = "Stable Fast 3D first, then Hunyuan3D Turbo"
        route = "路线 B+：显存高于 12GB 但不足 24GB，建议先试 Stable Fast 3D，再评估 Hunyuan3D Turbo/mini。"
    else:
        recommended_backend = "cloud_or_keep_triposr_baseline"
        route = "路线 C：显存低于 8GB，不建议本机尝试高质量后端，建议云端生成展示模型或继续 TripoSR baseline。"

    if free_gb < 40:
        warnings.append("磁盘剩余空间低于 40GB，不建议下载或展开大型 3D 生成模型权重。")
    elif free_gb < 100:
        warnings.append("磁盘空间可做轻量试验，但 Hunyuan3D-2.1 完整权重和输出可能较紧张。")

    if not torch_info.get("available"):
        warnings.append("当前项目环境未检测到 PyTorch，后续试跑新后端前需要单独准备依赖环境。")
    elif not torch_info.get("cuda_available") and cuda_available:
        warnings.append("检测到 GPU，但当前 PyTorch CUDA 不可用，可能需要新建独立生成环境。")

    return {
        "recommended_backend": recommended_backend,
        "recommended_route": route,
        "max_vram_gb": max_vram,
        "cuda_available": cuda_available,
        "disk_enough_for_large_models": free_gb >= 100,
        "disk_enough_for_small_trial": free_gb >= 40,
        "warnings": warnings,
    }


def format_bool(value):
    return "是" if value else "否"


def build_markdown(data):
    gpus = data["gpu"]["gpus"]
    gpu_text = ", ".join(
        f"{gpu.get('name', 'unknown')} ({gpu.get('vram_gb', 'unknown')} GB)"
        for gpu in gpus
    ) or "未检测到"

    torch_info = data["pytorch"]
    tools = data["tools"]
    rec = data["recommendation"]
    disk = data["disk"]

    warnings = rec["warnings"] or ["未发现阻塞性风险，但仍建议先做 3～5 张 ROI 的最小对比实验。"]
    warning_lines = "\n".join(f"- {item}" for item in warnings)

    return f"""# 3D 生成后端替换环境预检报告

## 1. 检查目的

本检查用于判断当前机器是否适合尝试更高质量的 image-to-3D 后端，例如 Hunyuan3D-2.1、Hunyuan3D-2.0 Turbo / mini 或 Stable Fast 3D。

本轮只做环境预检，不下载模型、不安装依赖、不运行 Hunyuan3D / Stable Fast 3D、不修改 plus.py、不调用 Qwen、不调用 TripoSR、不重跑缓存实验。

## 2. 本机环境

- 检查时间：{data["checked_at"]}
- 操作系统：{data["system"]["platform"]}
- Python 版本：{data["python"]["version"]}
- Python 可执行文件：{data["python"]["executable"]}
- 当前项目虚拟环境路径：{data["python"]["virtual_env"]}
- GPU：{gpu_text}
- nvidia-smi：{format_bool(data["nvidia_smi"]["available"])}
- CUDA 是否可用：{format_bool(rec["cuda_available"])}
- PyTorch 是否可用：{format_bool(torch_info["available"])}
- PyTorch 版本：{torch_info.get("version") or "未检测到"}
- PyTorch CUDA 可用：{format_bool(torch_info.get("cuda_available"))}
- PyTorch CUDA 版本：{torch_info.get("cuda_version") or "未检测到"}
- 磁盘总空间：{disk["total_gb"]} GB
- 磁盘剩余空间：{disk["free_gb"]} GB
- git：{format_bool(tools["git"]["available"])} {tools["git"].get("version") or ""}
- cmake：{format_bool(tools["cmake"]["available"])} {tools["cmake"].get("version") or ""}
- ninja：{format_bool(tools["ninja"]["available"])} {tools["ninja"].get("version") or ""}

## 3. 后端适配建议

- 最大检测显存：{rec["max_vram_gb"]} GB
- 推荐后端：{rec["recommended_backend"]}
- 推荐路线：{rec["recommended_route"]}

显存判断规则：

- VRAM >= 24GB：推荐优先 Hunyuan3D-2.1；
- VRAM 8GB～12GB：推荐 Stable Fast 3D 或 Hunyuan3D Turbo / mini；
- VRAM < 8GB：不建议本地尝试高质量后端，建议云端或继续 TripoSR baseline。

## 4. 风险说明

- Hunyuan3D-2.1 可能显存需求较高，完整 shape + texture 流程对本地硬件压力较大；
- Stable Fast 3D 与 TripoSR 路线更接近，替换成本可能较低；
- 当前不应直接替换 plus.py；
- 后续应通过 generator_adapter.py 统一适配不同生成后端；
- TripoSR 应继续保留为 baseline 和 fallback；
- 新后端试跑前应先选 3～5 张 ROI 做最小对比实验。

当前风险提示：

{warning_lines}

## 5. 下一步建议

{rec["recommended_route"]}

建议下一步只做离线最小实验：

1. 选取 3～5 张 ROI 图片；
2. 使用 TripoSR 和候选新后端分别生成；
3. 填写 alternative_generator_eval_template.csv；
4. 比较 generation_success、generation_time_ms、recognizable、suitable_for_demo、glb_path；
5. 再决定是否新增 generator_adapter.py。
"""


def main():
    parser = argparse.ArgumentParser(description="Check local environment for alternative 3D generators.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for markdown/json reports.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nvidia_info = detect_nvidia_smi()
    torch_info = detect_torch()
    gpus = choose_gpu_source(nvidia_info, torch_info)
    disk_info = get_disk_info(Path.cwd())
    recommendation = make_recommendation(gpus, torch_info, disk_info)

    data = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", ""),
            "virtual_env": os.environ.get("VIRTUAL_ENV") or sys.prefix,
        },
        "nvidia_smi": nvidia_info,
        "pytorch": torch_info,
        "gpu": {
            "gpus": gpus,
            "gpu_count": len(gpus),
        },
        "disk": disk_info,
        "tools": {
            "git": detect_tool("git"),
            "cmake": detect_tool("cmake"),
            "ninja": detect_tool("ninja", ["--version"]),
        },
        "recommendation": recommendation,
    }

    md_path = output_dir / "alternative_generator_env_check.md"
    json_path = output_dir / "alternative_generator_env_check.json"
    md_path.write_text(build_markdown(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    gpu_names = ", ".join(gpu.get("name", "unknown") for gpu in gpus) or "未检测到"
    vram_text = (
        f"{recommendation['max_vram_gb']} GB" if recommendation["max_vram_gb"] else "未检测到"
    )

    print("=" * 72)
    print(f"GPU 型号: {gpu_names}")
    print(f"VRAM: {vram_text}")
    print(f"CUDA 是否可用: {format_bool(recommendation['cuda_available'])}")
    print(f"推荐后端: {recommendation['recommended_backend']}")
    print(f"alternative_generator_env_check.md: {md_path}")
    print(f"alternative_generator_env_check.json: {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
