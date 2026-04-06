"""BLEU and ROUGE evaluation metrics for translation quality."""

from typing import List


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (simplified version without corpus-level stats).
    Returns a value between 0 and 1.

    Args:
        reference: reference translation (ground truth)
        hypothesis: generated translation
        max_n: maximum n-gram size

    Returns:
        BLEU score
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Calculate n-gram matches
    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        total = max(1, len(hyp_ngrams))
        scores.append(matches / total)

    # Brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        brevity_penalty = (
            (len(hyp_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0
        )
    else:
        brevity_penalty = 1.0

    # Geometric mean of n-gram scores
    import math

    geometric_mean = math.exp(sum(math.log(s) if s > 0 else 0 for s in scores) / len(scores))
    bleu = brevity_penalty * geometric_mean

    return min(1.0, max(0.0, bleu))


def rouge_score(reference: str, hypothesis: str) -> dict:
    """
    Calculate ROUGE-L (Longest Common Subsequence) score.

    Args:
        reference: reference translation
        hypothesis: generated translation

    Returns:
        Dict with precision, recall, and f-score
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0

    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    return {"precision": precision, "recall": recall, "f_score": f_score}


def _get_ngrams(tokens: List[str], n: int):
    """Extract n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)
    return ngrams


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def evaluate_translation_pair(reference: str, hypothesis: str) -> dict:
    """
    Evaluate a translation pair using multiple metrics.

    Args:
        reference: reference translation
        hypothesis: generated translation

    Returns:
        Dict with BLEU, ROUGE-L scores and length info
    """
    bleu = bleu_score(reference, hypothesis)
    rouge = rouge_score(reference, hypothesis)

    return {
        "bleu": round(bleu, 4),
        "rouge_precision": round(rouge["precision"], 4),
        "rouge_recall": round(rouge["recall"], 4),
        "rouge_f_score": round(rouge["f_score"], 4),
        "reference_length": len(reference.split()),
        "hypothesis_length": len(hypothesis.split()),
    }


def batch_evaluate(references: List[str], hypotheses: List[str]) -> dict:
    """
    Evaluate a batch of translation pairs.

    Args:
        references: list of reference translations
        hypotheses: list of generated translations

    Returns:
        Dict with average scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")

    results = [
        evaluate_translation_pair(ref, hyp) for ref, hyp in zip(references, hypotheses)
    ]

    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_rouge_f = sum(r["rouge_f_score"] for r in results) / len(results)

    return {
        "average_bleu": round(avg_bleu, 4),
        "average_rouge_f": round(avg_rouge_f, 4),
        "num_samples": len(results),
        "individual_scores": results,
    }
