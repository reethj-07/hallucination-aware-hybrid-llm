"""Full RAG evaluation with RAGAS metrics and MLflow logging."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mlflow
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from api.config import get_settings
from rag.pipeline import get_pipeline


def compute_em(prediction: str, ground_truths: list[str]) -> float:
    pred = prediction.strip().lower()
    return float(any(pred == gt.strip().lower() for gt in ground_truths))


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    def _f1(pred_tokens: set[str], gt_tokens: set[str]) -> float:
        if not pred_tokens or not gt_tokens:
            return 0.0
        tp = len(pred_tokens & gt_tokens)
        return 2 * tp / (len(pred_tokens) + len(gt_tokens))

    pred_tokens = set(prediction.lower().split())
    return max((_f1(pred_tokens, set(gt.lower().split())) for gt in ground_truths), default=0.0)


def config_hash(settings: Any) -> str:
    config_str = json.dumps(
        {
            "model": settings.rag_embed_model,
            "top_k": settings.rag_top_k,
            "lightweight": settings.rag_lightweight,
            "skip_chunking": settings.rag_skip_chunking,
            "rerank": settings.rag_rerank,
            "faithfulness_threshold": settings.faithfulness_threshold,
        },
        sort_keys=True,
    )
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]


def main(eval_path: str, use_mlflow: bool = True) -> None:
    settings = get_settings()
    pairs = [json.loads(line) for line in Path(eval_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    pipeline = get_pipeline(settings)

    results: list[dict[str, Any]] = []
    for pair in pairs:
        question = pair.get("question", pair.get("query", ""))
        ground_truths = pair.get("expected_answers", [])
        if not ground_truths and "answer" in pair:
            ground_truths = [pair["answer"]]

        result = asyncio.run(pipeline.arun(question, top_k=settings.rag_top_k))
        results.append(
            {
                "question": question,
                "answer": result.answer,
                "contexts": result.retrieved_documents,
                "ground_truths": ground_truths,
                "faithfulness_score": result.faithfulness_score,
                "em": compute_em(result.answer, ground_truths),
                "f1": compute_f1(result.answer, ground_truths),
            }
        )

    em = sum(row["em"] for row in results) / len(results)
    f1 = sum(row["f1"] for row in results) / len(results)
    avg_faith = sum(row["faithfulness_score"] for row in results) / len(results)

    ragas_dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truths"][0] if row["ground_truths"] else "",
            }
            for row in results
        ]
    )

    ragas_result = ragas_evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "em": round(em, 4),
        "f1": round(f1, 4),
        "avg_nli_faithfulness": round(avg_faith, 4),
        "ragas_faithfulness": round(float(ragas_result["faithfulness"]), 4),
        "ragas_answer_relevancy": round(float(ragas_result["answer_relevancy"]), 4),
        "ragas_context_precision": round(float(ragas_result["context_precision"]), 4),
        "ragas_context_recall": round(float(ragas_result["context_recall"]), 4),
        "n_samples": len(results),
    }

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if use_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"eval_{config_hash(settings)}"):
            mlflow.log_params(
                {
                    "embed_model": settings.rag_embed_model,
                    "top_k": settings.rag_top_k,
                    "lightweight": settings.rag_lightweight,
                    "skip_chunking": settings.rag_skip_chunking,
                    "faithfulness_threshold": settings.faithfulness_threshold,
                    "eval_set": eval_path,
                    "n_samples": len(results),
                }
            )
            mlflow.log_metrics(metrics)

    output_path = Path("results") / f"eval_{config_hash(settings)}_{int(time.time())}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps({"config": config_hash(settings), "metrics": metrics, "per_sample": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval/qa_pairs_v2.jsonl")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    main(args.eval_set, use_mlflow=not args.no_mlflow)
