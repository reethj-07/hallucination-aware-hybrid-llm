"""Ablation study for hallucination guard layers."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

import mlflow

from api.config import get_settings
from rag.hallucination_guard import ABSTENTION_STRING, verify_answer
from rag.pipeline import get_pipeline

GuardMode = Literal["none", "retrieval_only", "retrieval_prompt", "full"]


def compute_em(prediction: str, ground_truths: list[str]) -> float:
    pred = prediction.strip().lower()
    return float(any(pred == gt.strip().lower() for gt in ground_truths))


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    pred_tokens = set(prediction.lower().split())
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gt in ground_truths:
        gt_tokens = set(gt.lower().split())
        if not gt_tokens:
            continue
        overlap = len(pred_tokens & gt_tokens)
        score = (2 * overlap) / (len(pred_tokens) + len(gt_tokens))
        best = max(best, score)
    return best


async def run_condition(question: str, mode: GuardMode):
    settings = get_settings()
    pipeline = get_pipeline(settings)

    if mode == "none":
        generated = await pipeline.generate(question, "")
        return {
            "answer": generated,
            "contexts": [],
            "faithfulness_score": 0.0,
            "guard_method": "none",
        }

    embedding = pipeline.embed_query(question)
    docs_meta, _scores = pipeline.retrieve(embedding, settings.rag_top_k)
    contexts = [str(doc.get("text", "")) for doc in docs_meta]
    context = "\n\n".join(contexts)

    if mode == "retrieval_only":
        generated = await pipeline.generate(question, context)
        return {
            "answer": generated,
            "contexts": contexts,
            "faithfulness_score": 0.0,
            "guard_method": "retrieval_only",
        }

    if mode == "retrieval_prompt":
        generated = await pipeline.generate(question, context)
        return {
            "answer": generated,
            "contexts": contexts,
            "faithfulness_score": 0.0,
            "guard_method": "retrieval_prompt",
        }

    generated = await pipeline.generate(question, context)
    answer, score, method = verify_answer(generated, contexts, use_nli=settings.use_nli_guard)
    return {
        "answer": answer,
        "contexts": contexts,
        "faithfulness_score": score,
        "guard_method": method,
    }


def run_ablation(eval_set: Path, use_mlflow: bool = True) -> None:
    pairs = [json.loads(line) for line in eval_set.read_text(encoding="utf-8").splitlines() if line.strip()]

    settings = get_settings()
    if use_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("ablation_guard_layers")

    for mode in ("none", "retrieval_only", "retrieval_prompt", "full"):
        rows = []
        for pair in pairs:
            question = pair.get("question", pair.get("query", ""))
            expected_answers = pair.get("expected_answers", [])
            if not expected_answers and "answer" in pair:
                expected_answers = [pair["answer"]]

            result = asyncio.run(run_condition(question, mode))
            answer = result["answer"]

            rows.append(
                {
                    "em": compute_em(answer, expected_answers),
                    "f1": compute_f1(answer, expected_answers),
                    "abstained": answer == ABSTENTION_STRING,
                    "hallucinated": bool(result["contexts"]) and answer != ABSTENTION_STRING and not compute_em(answer, expected_answers),
                    "faithfulness_score": float(result["faithfulness_score"]),
                }
            )

        metrics = {
            "em": sum(r["em"] for r in rows) / len(rows),
            "f1": sum(r["f1"] for r in rows) / len(rows),
            "hallucination_rate": sum(float(r["hallucinated"]) for r in rows) / len(rows),
            "abstention_rate": sum(float(r["abstained"]) for r in rows) / len(rows),
            "avg_faithfulness": sum(r["faithfulness_score"] for r in rows) / len(rows),
        }

        print(f"\nCondition {mode}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        if use_mlflow:
            with mlflow.start_run(run_name=f"ablation_{mode}"):
                mlflow.log_param("guard_mode", mode)
                mlflow.log_param("n_samples", len(rows))
                mlflow.log_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval/qa_pairs_v2.jsonl")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run_ablation(Path(args.eval_set), use_mlflow=not args.no_mlflow)
