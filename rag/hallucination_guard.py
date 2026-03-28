"""Multi-layer hallucination guard."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

from sentence_transformers import CrossEncoder

NLI_MODEL_ID = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.4"))
ABSTENTION_STRING = "Not found in retrieved documents"
STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "or",
        "it",
        "its",
        "be",
        "this",
        "that",
        "for",
        "with",
        "as",
        "by",
        "from",
    ]
)


@lru_cache(maxsize=1)
def _get_nli_model() -> Optional[CrossEncoder]:
    try:
        return CrossEncoder(NLI_MODEL_ID)
    except Exception:
        return None


def _compute_token_overlap(answer: str, context: str) -> float:
    def tokenize(text: str) -> set[str]:
        tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
        return tokens - STOPWORDS

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    if not answer_tokens:
        return 0.0
    intersection = answer_tokens & context_tokens
    union = answer_tokens | context_tokens
    return len(intersection) / len(union) if union else 0.0


def compute_token_overlap(answer: str, context: str) -> float:
    """Jaccard overlap after lowercasing and stopword removal."""
    return _compute_token_overlap(answer, context)


def compute_nli_faithfulness(answer: str, context: str) -> float:
    """Returns entailment probability in [0, 1]."""
    model = _get_nli_model()
    if model is None:
        return _compute_token_overlap(answer, context)
    scores = model.predict([(context, answer)], apply_softmax=True)[0]
    return float(scores[1])


def verify_answer(
    answer: str,
    retrieved_docs: list[str],
    use_nli: bool = True,
) -> tuple[str, float, str]:
    """Returns (final_answer, faithfulness_score, guard_method_used)."""
    if answer == ABSTENTION_STRING:
        return answer, 1.0, "abstention_passthrough"

    context = "\n".join(retrieved_docs)

    if use_nli:
        score = compute_nli_faithfulness(answer, context)
        method = "nli_entailment"
    else:
        score = compute_token_overlap(answer, context)
        method = "token_overlap"

    if score < FAITHFULNESS_THRESHOLD:
        return ABSTENTION_STRING, score, f"{method}_abstained"

    return answer, score, method
