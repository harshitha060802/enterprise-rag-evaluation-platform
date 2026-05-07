import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evals" / "golden_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "evals" / "retrieval_report.md"

sys.path.append(str(SRC_DIR))

from hybrid_search import get_bm25_results, get_semantic_results, reciprocal_rank_fusion
from rerank_search import rerank


def evaluate_question(question, expected_source_hint):
    semantic_results = get_semantic_results(question, top_k=8)
    bm25_results = get_bm25_results(question, top_k=8)

    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:8]

    candidate_docs = [item["doc"] for item in fused_results]
    reranked_results = rerank(question, candidate_docs)[:5]

    retrieved_source_ids = [
        doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))
        for doc, score in reranked_results
    ]

    passed = any(
        expected_source_hint in source_id
        for source_id in retrieved_source_ids
    )

    return passed, retrieved_source_ids


def main():
    total = 0
    passed_count = 0
    report_lines = [
        "# Retrieval Evaluation Report",
        "",
    ]


    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            total += 1
            question = row["question"]
            expected_source_hint = row["expected_source_hint"]

            passed, retrieved_source_ids = evaluate_question(
                question,
                expected_source_hint,
            )

            if passed:
                passed_count += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"{status}: {question}")
            print(f"Expected source hint: {expected_source_hint}")
            print("Retrieved sources:")
            for source_id in retrieved_source_ids:
                print(f"- {source_id}")
            print()
            report_lines.append(f"## {status}: {question}")
            report_lines.append("")
            report_lines.append(f"Expected source hint: `{expected_source_hint}`")
            report_lines.append("")
            report_lines.append("Retrieved sources:")
            report_lines.extend([f"- `{source_id}`" for source_id in retrieved_source_ids])
            report_lines.append("")


    score = passed_count / total if total else 0

    print(f"Retrieval score: {score:.2f} ({passed_count}/{total})")
    report_lines.append(f"## Final Score")
    report_lines.append("")
    report_lines.append(f"Retrieval score: `{score:.2f}` ({passed_count}/{total})")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


    if score < 0.80:
        raise SystemExit("Retrieval evaluation failed.")


if __name__ == "__main__":
    main()
