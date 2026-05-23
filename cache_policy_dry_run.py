from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from cache_policy_loader import decide_cache_policy, load_cache_policy
from cache_similarity import build_similarity_index, score_cache_entries


DEFAULT_POLICY_PATH = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/"
    "policy_simulation/cache_policy_v3_real_70.json"
)
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/policy_dry_run")
DEFAULT_CACHE_DIR = Path("runtime_assets/model_cache")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_query_image(query_image: str) -> Optional[Path]:
    if not query_image:
        return None
    path = Path(query_image)
    if not path.exists():
        raise FileNotFoundError(f"query_image does not exist: {path}")
    return path


def build_entries(cache_dir: Path, output_dir: Path):
    index_path = output_dir / "cache_similarity_index_dry_run.json"
    return build_similarity_index(
        cache_dir=cache_dir,
        reference_dir=cache_dir / "reference_images",
        output_path=index_path,
    )


def decision_explanation(policy_decision: str) -> str:
    if policy_decision == "auto_hit":
        return "当前样本可自动复用缓存模型，但本脚本不实际加载模型。"
    if policy_decision == "review":
        return "当前样本进入 review 区，未来前端应提示用户确认是否复用。"
    return "当前样本不复用缓存，未来应回退到原生成流程。"


def run_dry_run(
    query_text: str,
    query_image: Optional[Path],
    cache_dir: Path,
    policy_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_cache_policy(policy_path)
    fallback_reason = ""
    best = None
    entries = []
    results = []

    try:
        entries = build_entries(cache_dir, output_dir)
        if not entries:
            fallback_reason = "no_cache_entries"
        else:
            results = score_cache_entries(
                entries,
                query_text=query_text,
                query_image=query_image,
                text_weight=float(policy.get("text_weight", 0.5)),
                image_weight=float(policy.get("image_weight", 0.5)),
                threshold=float(policy.get("weak_threshold", 0.7)),
            )
            best = results[0] if results else None
            if best is None:
                fallback_reason = "no_similarity_result"
    except Exception as exc:
        fallback_reason = f"similarity_failed: {exc}"

    text_score = best.text_score if best else None
    image_score = best.image_score if best else None
    decision = decide_cache_policy(text_score, image_score, policy)
    best_model_path = best.model_path if best else ""

    if best_model_path and not Path(best_model_path).exists():
        fallback_reason = "best_model_path_missing"

    result = {
        "query_text": query_text,
        "query_image": str(query_image) if query_image else "",
        "text_score": text_score,
        "image_score": image_score,
        "policy_score": decision["policy_score"],
        "policy_decision": decision["policy_decision"],
        "weak_threshold": decision["weak_threshold"],
        "strong_threshold": decision["strong_threshold"],
        "best_keyword": best.keyword if best else "",
        "best_filename": best.filename if best else "",
        "best_model_path": best_model_path,
        "fallback_reason": fallback_reason,
        "cache_dir": str(cache_dir),
        "policy_path": str(policy_path),
        "cache_entry_count": len(entries),
    }

    write_json(output_dir / "dry_run_result.json", result)
    write_report(output_dir / "dry_run_report.md", result)

    if results:
        write_json(
            output_dir / "dry_run_all_candidates.json",
            {"results": [asdict(item) for item in results]},
        )

    return result


def write_report(path: Path, result: Dict[str, Any]) -> None:
    decision = result["policy_decision"]
    text = f"""# 缓存策略 dry-run 测试报告

## 输入

- query_text: {result.get('query_text') or 'N/A'}
- query_image: {result.get('query_image') or 'N/A'}
- cache_dir: {result.get('cache_dir')}
- policy_path: {result.get('policy_path')}

## 最佳缓存候选

- best_keyword: {result.get('best_keyword') or 'N/A'}
- best_filename: {result.get('best_filename') or 'N/A'}
- best_model_path: {result.get('best_model_path') or 'N/A'}
- best_model_path 是否存在: {bool(result.get('best_model_path') and Path(str(result.get('best_model_path'))).exists())}

## 分数与决策

- text_score: {result.get('text_score')}
- image_score: {result.get('image_score')}
- policy_score: {result.get('policy_score')}
- policy_decision: {decision}
- 是否可 auto_hit: {decision == 'auto_hit'}
- 是否需要 review: {decision == 'review'}
- 是否 miss: {decision == 'miss'}
- fallback_reason: {result.get('fallback_reason') or 'N/A'}

## 决策解释

{decision_explanation(decision)}

## 注意

本脚本只进行 dry-run 判断，不加载模型、不生成模型、不修改 `plus.py`，也不接入前端。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run cache policy decision without touching plus.py.")
    parser.add_argument("--query-text", default="")
    parser.add_argument("--query-image", default="")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    query_image = resolve_query_image(args.query_image)
    result = run_dry_run(
        query_text=args.query_text,
        query_image=query_image,
        cache_dir=Path(args.cache_dir),
        policy_path=Path(args.policy_path),
        output_dir=Path(args.output_dir),
    )

    print("=" * 72)
    print(f"dry_run_result.json: {Path(args.output_dir) / 'dry_run_result.json'}")
    print(f"dry_run_report.md: {Path(args.output_dir) / 'dry_run_report.md'}")
    print(f"policy_decision: {result['policy_decision']}")
    print(f"policy_score: {result['policy_score']}")
    print(f"best_model_path: {result['best_model_path']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
