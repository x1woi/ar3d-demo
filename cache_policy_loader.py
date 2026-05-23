from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_POLICY_PATH = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/"
    "policy_simulation/cache_policy_v3_real_70.json"
)


def load_cache_policy(policy_path: str | Path = DEFAULT_POLICY_PATH) -> Dict[str, Any]:
    path = Path(policy_path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_policy_score(
    text_score: Any,
    image_score: Any,
    text_weight: float,
    image_weight: float,
) -> float:
    text = _to_float(text_score)
    image = _to_float(image_score)
    if text is None and image is None:
        return 0.0
    if image is None:
        image = text
    if text is None:
        text = image
    return float(text_weight) * float(text) + float(image_weight) * float(image)


def decide_cache_policy(
    text_score: Any,
    image_score: Any,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    text_weight = float(policy.get("text_weight", 0.5))
    image_weight = float(policy.get("image_weight", 0.5))
    weak_threshold = float(policy.get("weak_threshold", 0.7))
    strong_threshold = float(policy.get("strong_threshold", 0.78))
    score = compute_policy_score(text_score, image_score, text_weight, image_weight)

    if score >= strong_threshold:
        decision = "auto_hit"
    elif score >= weak_threshold:
        decision = "review"
    else:
        decision = "miss"

    return {
        "policy_score": round(score, 6),
        "policy_decision": decision,
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "text_weight": text_weight,
        "image_weight": image_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load cache policy and decide cache reuse action.")
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--text-score", type=float, default=None)
    parser.add_argument("--image-score", type=float, default=None)
    args = parser.parse_args()

    policy = load_cache_policy(args.policy_path)
    decision = decide_cache_policy(args.text_score, args.image_score, policy)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
