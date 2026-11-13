
"""Evaluate retrieved evidence and preserve per-query traces."""

import re
from collections import defaultdict
from typing import Any


def _normalize(text: str) -> str:
    """Normalize whitespace for reliable evidence comparison."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_evidence_hit(
    question: dict[str, Any],
    hit: dict[str, Any],
) -> bool:
    """Check whether a retrieved chunk matches the gold source, page and quote."""
    if hit.get("document_id") != question.get("gold_source_id"):
        return False

    if int(hit.get("page", -1)) != int(question.get("gold_page", -2)):
        return False

    quote = _normalize(question.get("evidence_quote", ""))
    content = _normalize(hit.get("text", hit.get("content", "")))

    if not quote or not content:
        return False

    return quote in content


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate top-k retrieval and retain detailed evidence traces."""
    if not isinstance(question, dict):
        raise TypeError("question must be a dictionary.")

    if not isinstance(hits, list):
        raise TypeError("hits must be a list.")

    if not question.get("id"):
        raise ValueError("Question must have an id.")

    required = [
        "gold_source_id",
        "gold_page",
        "evidence_quote",
        "split",
    ]
    missing = [key for key in required if key not in question]

    if missing:
        raise ValueError(f"Question is missing fields: {missing}")

    checked_hits = []

    for rank, hit in enumerate(hits[:3], start=1):
        is_hit = _is_evidence_hit(question, hit)

        checked_hits.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "source": hit.get("source"),
            "page": hit.get("page"),
            "text": hit.get("text", hit.get("content", "")),
            "score": float(hit["score"]) if hit.get("score") is not None else None,
            "metadata": hit.get("metadata", {}),
            "evidence_hit": is_hit,
        })

    hit1 = bool(checked_hits and checked_hits[0]["evidence_hit"])
    hit3 = any(item["evidence_hit"] for item in checked_hits)

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question.get("question", ""),
        "expected_answer": question.get("expected_answer", ""),
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "filter": question.get("filter", {}),
        "hit_at_1": hit1,
        "evidence_hit_at_3": hit3,
        "retrieved_count": len(checked_hits),
        "hits": checked_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split.

    Each trace must contain a strategy field in addition to the fields
    returned by evaluate_query().
    """
    groups = defaultdict(list)

    for trace in traces:
        strategy = trace.get("strategy")
        split = trace.get("split")

        if not strategy or split not in {"dev", "test"}:
            raise ValueError(
                "Every trace must have a strategy and split='dev' or 'test'."
            )

        groups[(strategy, split)].append(trace)

    summary = {}

    for (strategy, split), items in sorted(groups.items()):
        total = len(items)
        hit1_count = sum(bool(item["hit_at_1"]) for item in items)
        hit3_count = sum(bool(item["evidence_hit_at_3"]) for item in items)

        summary.setdefault(strategy, {})[split] = {
            "question_count": total,
            "hit_at_1": {
                "hits": hit1_count,
                "total": total,
                "rate": hit1_count / total if total else 0.0,
            },
            "evidence_hit_at_3": {
                "hits": hit3_count,
                "total": total,
                "rate": hit3_count / total if total else 0.0,
            },
        }

    return summary
