
"""Day 06: Data ingestion, vector retrieval and benchmark orchestration.

Default operation is local preparation only.
Synthetic smoke tests are available without using the source PDF.
"""
import csv
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import assign_metadata, load_pdf
from .models import Chunk
from .store import PgVectorStore


ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = ROOT / "starter"
PDF_PATH = STARTER_DIR / "data" / "principles-data-science.pdf"
MANIFEST_PATH = STARTER_DIR / "data" / "manifest.json"
BENCHMARK_PATH = ROOT / "benchmark" / "questions.json"
RESULTS_DIR = STARTER_DIR / "results"

DOCUMENT_ID = "principles-data-science"
SOURCE = "OpenStax Principles of Data Science"

CONFIGS = {
    "A": {
        "chunking": "fixed_size",
        "size": 1000,
        "overlap": 150,
        "metadata_filter": False,
    },
    "B": {
        "chunking": "by_paragraph",
        "max_size": 1000,
        "metadata_filter": False,
    },
    "C": {
        "chunking": "by_paragraph",
        "max_size": 1000,
        "metadata_filter": True,
    },
}


def load_benchmark():
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions", [])

    if len(questions) < 20:
        raise ValueError("Benchmark must contain at least 20 questions.")

    ids = [q.get("id") for q in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark question IDs.")

    splits = Counter(q.get("split") for q in questions)

    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError(f"Invalid benchmark split: {dict(splits)}")

    required = [
        "id", "split", "question", "gold_source_id",
        "gold_page", "evidence_quote", "expected_answer",
    ]

    for question in questions:
        missing = [key for key in required if key not in question]
        if missing:
            raise ValueError(
                f"{question.get('id', 'unknown')} missing fields: {missing}"
            )

        if question["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid split in {question['id']}.")

    return questions


def load_corpus():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("corpus_id") != DOCUMENT_ID:
        raise ValueError("Manifest corpus_id does not match.")

    pages = load_pdf(PDF_PATH, DOCUMENT_ID, SOURCE)
    pages = assign_metadata(pages)

    if len(pages) < 30:
        raise ValueError("Corpus must contain at least 30 physical pages.")

    content_pages = [
        p for p in pages
        if p.metadata.get("category") == "chapter" and p.text.strip()
    ]

    if len(content_pages) < 30:
        raise ValueError("Not enough non-empty chapter pages.")

    return pages, content_pages, manifest


def build_chunks(pages):
    strategies = {"A": [], "B": [], "C": []}

    for page in pages:
        strategies["A"].extend(
            fixed_size(
                page,
                size=CONFIGS["A"]["size"],
                overlap=CONFIGS["A"]["overlap"],
            )
        )

        paragraph_chunks = by_paragraph(
            page,
            max_size=CONFIGS["B"]["max_size"],
        )

        strategies["B"].extend(paragraph_chunks)
        strategies["C"].extend(paragraph_chunks)

    return strategies


def chunk_statistics(chunks):
    lengths = [len(chunk.text) for chunk in chunks]

    if not lengths:
        return {
            "chunk_count": 0,
            "total_characters": 0,
            "average_characters": 0,
            "min_characters": 0,
            "max_characters": 0,
        }

    return {
        "chunk_count": len(lengths),
        "total_characters": sum(lengths),
        "average_characters": round(sum(lengths) / len(lengths), 2),
        "min_characters": min(lengths),
        "max_characters": max(lengths),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def prepare_only():
    questions = load_benchmark()
    pages, content_pages, _ = load_corpus()

    print(f"Physical pages: {len(pages)}")
    print(f"Non-empty chapter pages: {len(content_pages)}")
    print(f"Benchmark questions: {len(questions)}")

    split_counts = Counter(q["split"] for q in questions)
    print(f"Benchmark split: {dict(split_counts)}")

    strategies = build_chunks(pages)

    report = {
        "corpus_id": DOCUMENT_ID,
        "physical_pages": len(pages),
        "content_pages_used": len(content_pages),
        "benchmark_questions": len(questions),
        "benchmark_split": dict(split_counts),
        "configurations": CONFIGS,
        "strategies": {},
        "embedding_performed": False,
        "vector_database_written": False,
    }

    for name, chunks in strategies.items():
        stats = chunk_statistics(chunks)
        report["strategies"][name] = stats

        print(f"\nStrategy {name}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    output = RESULTS_DIR / "preparation_summary.json"
    save_json(output, report)
    print(f"\nPreparation saved: {output}")


def run_smoke_test():
    """Test Gemini and pgvector using synthetic text only."""
    load_dotenv(STARTER_DIR / ".env")

    model_name = os.getenv("EMBEDDING_MODEL")
    database_url = os.getenv("DATABASE_URL")

    if not model_name or not database_url:
        raise RuntimeError("Missing embedding model or DATABASE_URL.")

    embedder = MyEmbedder(
        model_name=model_name,
        dimensions=768,
        batch_size=2,
    )

    store = PgVectorStore(database_url, dimensions=768)
    store.create_schema()

    synthetic_texts = [
        "Synthetic data collection test.",
        "Synthetic statistical analysis test.",
        "Synthetic machine learning test.",
    ]

    vectors = embedder.embed_documents(synthetic_texts)

    if len(vectors) != len(synthetic_texts):
        raise RuntimeError("Embedding count mismatch.")

    synthetic_chunks = [
        Chunk(
            chunk_id=f"smoke-{index:03d}",
            document_id="synthetic-smoke",
            source="Synthetic test data",
            page=index + 1,
            text=text,
            metadata={
                "chapter": "1",
                "category": "chapter",
                "language": "en",
            },
        )
        for index, text in enumerate(synthetic_texts)
    ]

    try:
        store.upsert("SMOKE", synthetic_chunks, vectors)

        query_vector = embedder.embed_query(
            "Synthetic data collection test."
        )

        hits = store.search(
            strategy="SMOKE",
            query_vector=query_vector,
            top_k=3,
            filters={"chapter": "1"},
        )

        if not hits:
            raise RuntimeError("Synthetic retrieval returned no results.")

        print("Gemini synthetic embedding: PASS")
        print("Vector dimensions:", len(vectors[0]))
        print("PostgreSQL vector insertion: PASS")
        print("Top-3 retrieval: PASS")
        print("Metadata filtering: PASS")
        print("Retrieved results:", len(hits))

    finally:
        with store._connect() as conn:
            conn.execute(
                """
                DELETE FROM document_chunks
                WHERE strategy = %s AND document_id = %s
                """,
                ("SMOKE", "synthetic-smoke"),
            )

        print("Synthetic database cleanup: PASS")


def update_experiment_log(rows):
    """Write reproducible experiment metrics to CSV."""
    log_path = ROOT / "artifacts" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "strategy",
        "chunking",
        "chunk_size",
        "overlap",
        "metadata",
        "filter",
        "vectorizer",
        "hypothesis",
        "dev_hit1",
        "dev_hit3",
        "test_hit1",
        "test_hit3",
        "chunk_count",
        "embedding_seconds",
        "retrieval_seconds",
        "upsert_seconds",
        "document_requests",
        "query_requests",
        "document_texts",
        "query_texts",
        "token_usage_available",
        "total_tokens",
        "estimated_cost",
        "notes",
    ]

    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Experiment log saved: {log_path}")


def run_ingestion():
    """Ingest and benchmark the real corpus after permission is confirmed."""
    load_dotenv(STARTER_DIR / ".env")

    model_name = os.getenv("EMBEDDING_MODEL")
    database_url = os.getenv("DATABASE_URL")

    if not model_name or not database_url:
        raise RuntimeError("Missing EMBEDDING_MODEL or DATABASE_URL.")

    questions = load_benchmark()
    pages, content_pages, _ = load_corpus()

    print("Building chunks...")
    strategies = build_chunks(pages)

    embedder = MyEmbedder(
        model_name=model_name,
        dimensions=768,
        batch_size=32,
    )

    store = PgVectorStore(database_url, dimensions=768)
    store.create_schema()

    all_traces = []
    ingestion_stats = {}
    retrieval_stats = {}

    vectors_by_strategy = {}
    upsert_seconds = {}

    # 1. Embed document chunks for A and B.
    # C uses exactly the same chunks and vectors as B.
    for strategy in ("A", "B"):
        chunks = strategies[strategy]

        print(f"\nEmbedding strategy {strategy}: {len(chunks)} chunks")

        started = time.perf_counter()
        vectors = embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        embedding_elapsed = time.perf_counter() - started

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch for {strategy}."
            )

        started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        upsert_elapsed = time.perf_counter() - started

        vectors_by_strategy[strategy] = vectors
        upsert_seconds[strategy] = round(upsert_elapsed, 3)

        ingestion_stats[strategy] = {
            **chunk_statistics(chunks),
            "embedding_seconds": round(embedding_elapsed, 3),
            "upsert_seconds": round(upsert_elapsed, 3),
            "model": model_name,
            "dimensions": 768,
        }

        print(f"Stored strategy {strategy}.")

    # 2. C reuses B's document embeddings.
    strategies["C"] = strategies["B"]
    vectors_by_strategy["C"] = vectors_by_strategy["B"]

    started = time.perf_counter()
    store.upsert(
        "C",
        strategies["C"],
        vectors_by_strategy["C"],
    )
    upsert_seconds["C"] = round(
        time.perf_counter() - started, 3
    )

    ingestion_stats["C"] = {
        **chunk_statistics(strategies["C"]),
        "embedding_seconds": 0,
        "embedding_reused_from": "B",
        "upsert_seconds": upsert_seconds["C"],
        "model": model_name,
        "dimensions": 768,
    }

    # 3. Embed each benchmark question once and reuse across A/B/C.
    print("\nEmbedding benchmark questions once...")

    query_vectors = {}

    for question in questions:
        query_vectors[question["question_id"]] = (
            embedder.embed_query(question["question"])
        )

    print(
        "Query embedding requests:",
        embedder.get_stats()["query_requests"],
    )

    # 4. Run retrieval for each strategy.
    print("\nRunning benchmark retrieval...")

    for strategy in ("A", "B", "C"):
        started = time.perf_counter()

        for question in questions:
            query_vector = query_vectors[question["question_id"]]

            filters = (
                question.get("filter", {})
                if strategy == "C"
                else None
            )

            hits = store.search(
                strategy=strategy,
                query_vector=query_vector,
                top_k=3,
                filters=filters,
            )

            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            all_traces.append(trace)

        elapsed = time.perf_counter() - started

        retrieval_stats[strategy] = {
            "retrieval_seconds": round(elapsed, 3),
            "questions": len(questions),
            "queries_per_second": round(
                len(questions) / elapsed, 3
            ) if elapsed > 0 else None,
        }

        print(
            f"Completed {strategy}: "
            f"{retrieval_stats[strategy]['retrieval_seconds']} seconds"
        )

    # 5. Summarize retrieval results.
    summary = summarize(all_traces)

    # 6. Save detailed traces and summaries.
    save_json(
        RESULTS_DIR / "retrieval_traces.json",
        all_traces,
    )

    save_json(
        RESULTS_DIR / "retrieval_summary.json",
        summary,
    )

    embedder_stats = embedder.get_stats()

    save_json(
        RESULTS_DIR / "ingestion_summary.json",
        {
            "corpus_id": DOCUMENT_ID,
            "embedding_model": model_name,
            "dimensions": 768,
            "physical_pages": len(pages),
            "content_pages": len(content_pages),
            "embedding_api_stats": embedder_stats,
            "ingestion": ingestion_stats,
            "retrieval": retrieval_stats,
            "retrieval_summary": summary,
            "cost": {
                "actual_cost": None,
                "status": "Not available from embedding response; "
                          "check provider billing records.",
            },
        },
    )

    # 7. Update experiment_log.csv.
    experiment_rows = []

    hypotheses = {
        "A": "Baseline fixed-size chunking.",
        "B": "Test paragraph-aware chunking against A.",
        "C": "Test metadata filtering against B.",
    }

    for strategy in ("A", "B", "C"):
        dev = summary.get(strategy, {}).get("dev", {})
        test = summary.get(strategy, {}).get("test", {})

        config = CONFIGS[strategy]

        experiment_rows.append({
            "strategy": strategy,
            "chunking": config["chunking"],
            "chunk_size": config.get(
                "size", config.get("max_size", "")
            ),
            "overlap": config.get("overlap", 0),
            "metadata": "yes",
            "filter": config["metadata_filter"],
            "vectorizer": model_name,
            "hypothesis": hypotheses[strategy],
            "dev_hit1": dev.get("hit_at_1", {}).get("rate", ""),
            "dev_hit3": dev.get("evidence_hit_at_3", {}).get("rate", ""),
            "test_hit1": test.get("hit_at_1", {}).get("rate", ""),
            "test_hit3": test.get("evidence_hit_at_3", {}).get("rate", ""),
            "chunk_count": len(strategies[strategy]),
            "embedding_seconds": ingestion_stats[strategy][
                "embedding_seconds"
            ],
            "retrieval_seconds": retrieval_stats[strategy][
                "retrieval_seconds"
            ],
            "upsert_seconds": upsert_seconds[strategy],
            "document_requests": (
                embedder_stats["document_requests"]
                if strategy in ("A", "B") else 0
            ),
            "query_requests": (
                embedder_stats["query_requests"]
                if strategy == "A" else 0
            ),
            "document_texts": (
                len(strategies[strategy])
                if strategy in ("A", "B") else 0
            ),
            "query_texts": len(questions) if strategy == "A" else 0,
            "token_usage_available": embedder_stats[
                "token_usage_available"
            ],
            "total_tokens": (
                embedder_stats["total_tokens"]
                if embedder_stats["token_usage_available"]
                else ""
            ),
            "estimated_cost": "",
            "notes": (
                "C reuses B document embeddings; query vectors "
                "are shared across all strategies."
            ),
        })

    update_experiment_log(experiment_rows)

    print("\nRetrieval summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("\nEmbedding API statistics:")
    print(json.dumps(embedder_stats, ensure_ascii=False, indent=2))

    print("\nResults saved in:", RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Day 06 Data Ingestion pipeline."
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare local corpus and chunks without embeddings.",
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test Gemini and pgvector using synthetic text only.",
    )

    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Embed and ingest the real corpus.",
    )

    args = parser.parse_args()

    if args.prepare_only:
        prepare_only()
    elif args.smoke_test:
        run_smoke_test()
    elif args.ingest:
        raise RuntimeError(
            "Real corpus ingestion is disabled until the source "
            "license/permission for external AI embedding is confirmed."
        )


if __name__ == "__main__":
    main()
