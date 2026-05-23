from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CacheEntry:
    cache_id: str
    keyword: str
    filename: str
    model_path: str
    size_bytes: int
    source: str
    reference_image: str = ""
    split_report: str = ""


@dataclass
class MatchResult:
    cache_id: str
    keyword: str
    filename: str
    model_path: str
    reference_image: str
    text_score: float
    image_score: Optional[float]
    fused_score: float
    text_available: bool
    image_available: bool
    decision: str


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_text(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "").strip() if not ch.isspace())


def char_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> Dict[str, float]:
    text = safe_text(text)
    feats: Dict[str, float] = {}
    if not text:
        return feats
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for idx in range(0, len(text) - n + 1):
            token = text[idx : idx + n]
            feats[token] = feats.get(token, 0.0) + 1.0
    return feats


def sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def text_similarity(query: str, candidate: str) -> float:
    query = safe_text(query)
    candidate = safe_text(candidate)
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        shorter = min(len(query), len(candidate))
        longer = max(len(query), len(candidate))
        return max(0.75, shorter / max(1, longer))
    return sparse_cosine(char_ngrams(query), char_ngrams(candidate))


def read_image(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def normalized_histogram(values: np.ndarray, bins: int, value_range: Tuple[int, int]) -> np.ndarray:
    hist = cv2.calcHist([values], [0], None, [bins], value_range).reshape(-1)
    total = float(hist.sum())
    if total <= 1e-12:
        return np.zeros(bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def average_hash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (small > float(np.mean(small))).astype(np.float32).reshape(-1)


def image_signature(path: Path) -> Optional[np.ndarray]:
    img = read_image(path)
    if img is None:
        return None

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_hist = normalized_histogram(hsv[:, :, 0], 32, (0, 180))
    s_hist = normalized_histogram(hsv[:, :, 1], 16, (0, 256))
    v_hist = normalized_histogram(hsv[:, :, 2], 16, (0, 256))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([float(np.count_nonzero(edges)) / edges.size], dtype=np.float32)

    ahash = average_hash(gray, size=16)
    sig = np.concatenate([h_hist, s_hist, v_hist, edge_density, ahash]).astype(np.float32)
    norm = float(np.linalg.norm(sig))
    if norm <= 1e-12:
        return sig
    return sig / norm


def dense_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def image_similarity(query_image: Path, candidate_image: Path) -> Optional[float]:
    if not query_image.exists() or not candidate_image.exists():
        return None
    query_sig = image_signature(query_image)
    candidate_sig = image_signature(candidate_image)
    if query_sig is None or candidate_sig is None:
        return None
    return max(0.0, min(1.0, dense_cosine(query_sig, candidate_sig)))


def find_reference_image(reference_dir: Path, cache_id: str, keyword: str, filename: str) -> str:
    if not reference_dir.exists():
        return ""
    stems = {
        safe_text(cache_id),
        safe_text(keyword),
        safe_text(Path(filename).stem),
    }
    for path in reference_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = safe_text(path.stem)
        if stem in stems or any(item and item in stem for item in stems):
            return str(path)
    return ""


def stable_ascii_id(text: str) -> str:
    return hashlib.md5((text or "cache").encode("utf-8", errors="ignore")).hexdigest()[:12]


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buffer.tofile(str(path))


def build_similarity_index(
    cache_dir: Path,
    reference_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> List[CacheEntry]:
    cache_dir = Path(cache_dir)
    reference_dir = Path(reference_dir) if reference_dir else cache_dir / "reference_images"
    raw_index = read_json(cache_dir / "cache_index.json", {})
    entries: List[CacheEntry] = []

    for cache_id, item in raw_index.items():
        keyword = str(item.get("keyword") or cache_id)
        filename = str(item.get("filename") or "")
        if not filename:
            continue
        model_path = cache_dir / filename
        reference_image = str(item.get("reference_image") or "")
        if not reference_image:
            reference_image = find_reference_image(reference_dir, cache_id, keyword, filename)

        entries.append(
            CacheEntry(
                cache_id=str(cache_id),
                keyword=keyword,
                filename=filename,
                model_path=str(model_path),
                size_bytes=int(item.get("size_bytes") or (model_path.stat().st_size if model_path.exists() else 0)),
                source=str(item.get("source") or ""),
                reference_image=reference_image,
                split_report=str(item.get("split_report") or ""),
            )
        )

    payload = {
        "schema": "ar_cache_similarity_index.v1",
        "cache_dir": str(cache_dir),
        "reference_dir": str(reference_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [asdict(entry) for entry in entries],
    }
    if output_path:
        write_json(output_path, payload)
    return entries


def attach_reference_image(cache_dir: Path, keyword: str, reference_image: Path) -> Path:
    cache_dir = Path(cache_dir)
    reference_image = Path(reference_image)
    if not reference_image.exists():
        raise FileNotFoundError(f"Reference image does not exist: {reference_image}")

    raw_index = read_json(cache_dir / "cache_index.json", {})
    if not raw_index:
        raise FileNotFoundError(f"Cache index is empty or missing: {cache_dir / 'cache_index.json'}")

    target_key = ""
    keyword_norm = safe_text(keyword)
    for cache_id, item in raw_index.items():
        candidates = {
            safe_text(str(cache_id)),
            safe_text(str(item.get("keyword") or "")),
            safe_text(Path(str(item.get("filename") or "")).stem),
        }
        if keyword_norm in candidates or any(keyword_norm and keyword_norm in item for item in candidates):
            target_key = str(cache_id)
            break

    if not target_key:
        raise KeyError(f"No cache entry matched keyword: {keyword}")

    ref_dir = cache_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    suffix = reference_image.suffix.lower() or ".jpg"
    dst = ref_dir / f"ref_{stable_ascii_id(target_key + str(raw_index[target_key].get('filename', '')))}{suffix}"

    img = read_image(reference_image)
    if img is None:
        raise ValueError(f"Could not read image: {reference_image}")
    save_image(dst, img)

    raw_index[target_key]["reference_image"] = str(dst)
    raw_index[target_key]["reference_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(cache_dir / "cache_index.json", raw_index)
    return dst


def load_similarity_index(path: Path) -> List[CacheEntry]:
    payload = read_json(Path(path), {})
    entries = payload.get("entries", [])
    return [CacheEntry(**item) for item in entries]


def score_cache_entries(
    entries: Iterable[CacheEntry],
    query_text: str = "",
    query_image: Optional[Path] = None,
    text_weight: float = 0.45,
    image_weight: float = 0.55,
    threshold: float = 0.62,
) -> List[MatchResult]:
    results: List[MatchResult] = []

    for entry in entries:
        t_score = text_similarity(query_text, entry.keyword) if query_text else 0.0
        i_score = None
        if query_image and entry.reference_image:
            i_score = image_similarity(Path(query_image), Path(entry.reference_image))

        text_available = bool(query_text)
        image_available = i_score is not None

        weights = []
        score_parts = []
        if text_available:
            weights.append(text_weight)
            score_parts.append(text_weight * t_score)
        if image_available:
            weights.append(image_weight)
            score_parts.append(image_weight * float(i_score))

        fused = sum(score_parts) / sum(weights) if weights else 0.0
        decision = "hit" if fused >= threshold else "miss"

        results.append(
            MatchResult(
                cache_id=entry.cache_id,
                keyword=entry.keyword,
                filename=entry.filename,
                model_path=entry.model_path,
                reference_image=entry.reference_image,
                text_score=round(t_score, 4),
                image_score=round(i_score, 4) if i_score is not None else None,
                fused_score=round(fused, 4),
                text_available=text_available,
                image_available=image_available,
                decision=decision,
            )
        )

    results.sort(key=lambda item: item.fused_score, reverse=True)
    return results


def write_results(output_dir: Path, results: List[MatchResult], query: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ar_cache_similarity_experiment.v1",
        "query": query,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best": asdict(results[0]) if results else None,
        "results": [asdict(item) for item in results],
    }
    write_json(output_dir / "similarity_results.json", payload)

    with (output_dir / "similarity_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "decision",
                "cache_id",
                "keyword",
                "filename",
                "text_score",
                "image_score",
                "fused_score",
                "reference_image",
                "model_path",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(results, start=1):
            row = asdict(item)
            row["rank"] = rank
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def list_query_images(query_dir: Path) -> List[Path]:
    if not query_dir.exists():
        return []
    return sorted(
        path for path in query_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AR cache similarity index and hit experiment: text, image, and fused scoring."
    )
    parser.add_argument("--cache-dir", type=str, default="runtime_assets/model_cache")
    parser.add_argument("--reference-dir", type=str, default="")
    parser.add_argument("--index-path", type=str, default="runtime_assets/model_cache/cache_similarity_index.json")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--attach-keyword", type=str, default="")
    parser.add_argument("--attach-image", type=str, default="")
    parser.add_argument("--query-text", type=str, default="")
    parser.add_argument("--query-image", type=str, default="")
    parser.add_argument("--query-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="paper_repro_outputs/cache_similarity_test")
    parser.add_argument("--text-weight", type=float, default=0.45)
    parser.add_argument("--image-weight", type=float, default=0.55)
    parser.add_argument("--threshold", type=float, default=0.62)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    index_path = Path(args.index_path)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    if args.attach_keyword and args.attach_image:
        attached = attach_reference_image(cache_dir, args.attach_keyword, Path(args.attach_image))
        print(f"Reference image attached: {attached}")
        args.build_index = True

    if args.build_index or not index_path.exists():
        entries = build_similarity_index(cache_dir, reference_dir, index_path)
        print(f"Cache similarity index written: {index_path}")
    else:
        entries = load_similarity_index(index_path)

    output_dir = Path(args.output_dir)

    query_images = list_query_images(Path(args.query_dir)) if args.query_dir else []
    if args.query_image:
        query_images = [Path(args.query_image)]

    if query_images:
        for query_image in query_images:
            query_text = args.query_text or query_image.stem
            results = score_cache_entries(
                entries,
                query_text=query_text,
                query_image=query_image,
                text_weight=args.text_weight,
                image_weight=args.image_weight,
                threshold=args.threshold,
            )
            run_dir = output_dir / query_image.stem
            write_results(
                run_dir,
                results,
                {
                    "query_text": query_text,
                    "query_image": str(query_image),
                    "text_weight": args.text_weight,
                    "image_weight": args.image_weight,
                    "threshold": args.threshold,
                },
            )
            best = results[0] if results else None
            print(f"Query image: {query_image}")
            print(f"Best: {best.keyword if best else 'none'} score={best.fused_score if best else 0}")
    else:
        results = score_cache_entries(
            entries,
            query_text=args.query_text,
            query_image=None,
            text_weight=args.text_weight,
            image_weight=args.image_weight,
            threshold=args.threshold,
        )
        write_results(
            output_dir,
            results,
            {
                "query_text": args.query_text,
                "query_image": "",
                "text_weight": args.text_weight,
                "image_weight": args.image_weight,
                "threshold": args.threshold,
            },
        )
        best = results[0] if results else None
        print(f"Query text: {args.query_text}")
        print(f"Best: {best.keyword if best else 'none'} score={best.fused_score if best else 0}")


if __name__ == "__main__":
    main()
