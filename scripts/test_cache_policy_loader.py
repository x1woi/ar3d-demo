from __future__ import annotations

from cache_policy_loader import compute_policy_score, decide_cache_policy


POLICY = {
    "text_weight": 0.5,
    "image_weight": 0.5,
    "weak_threshold": 0.7,
    "strong_threshold": 0.78,
}


def assert_decision(text_score, image_score, expected: str) -> None:
    result = decide_cache_policy(text_score, image_score, POLICY)
    actual = result["policy_decision"]
    assert actual == expected, f"expected {expected}, got {actual}: {result}"


def run_tests() -> None:
    assert_decision(1.0, 1.0, "auto_hit")
    assert_decision(0.75, 0.75, "review")
    assert_decision(0.5, 0.5, "miss")

    assert compute_policy_score(0.8, None, 0.5, 0.5) == 0.8
    assert_decision(0.8, None, "auto_hit")

    assert compute_policy_score(None, 0.75, 0.5, 0.5) == 0.75
    assert_decision(None, 0.75, "review")

    assert compute_policy_score(None, None, 0.5, 0.5) == 0.0
    assert_decision(None, None, "miss")

    assert_decision(0.78, 0.78, "auto_hit")
    assert_decision(0.7, 0.7, "review")
    assert_decision(0.699, 0.699, "miss")


if __name__ == "__main__":
    run_tests()
    print("All cache_policy_loader tests passed.")
