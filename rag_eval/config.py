"""Metric definitions: targets, alarm thresholds, diagnosis and fixes (from the RAG Mastery Class 6 guide)."""
import math

# direction "higher": pass if score >= target, warn if >= limit, else fail
# direction "lower" : pass if score <= target, warn if <= limit, else fail
METRICS = {
    "context_precision": {
        "label": "Context Precision", "group": "Retrieval", "direction": "higher",
        "target": 0.80, "limit": 0.70,
        "asks": "Are the retrieved chunks actually relevant?",
        "diagnosis": "Retrieval is returning JUNK chunks.",
        "fix": "Reduce chunk size, try hybrid retrieval (BM25 + vector), add metadata filters, add reranking.",
    },
    "context_recall": {
        "label": "Context Recall", "group": "Retrieval", "direction": "higher",
        "target": 0.80, "limit": 0.70,
        "asks": "Were ALL relevant chunks found?",
        "diagnosis": "Relevant docs exist but are MISSED by retrieval.",
        "fix": "Increase k from 3 to 5-7, add hybrid retrieval, add query rewriting or HyDE.",
    },
    "faithfulness": {
        "label": "Faithfulness", "group": "Generation", "direction": "higher",
        "target": 0.90, "limit": 0.80,
        "asks": "Is the answer grounded in the retrieved context?",
        "diagnosis": "LLM is HALLUCINATING (adding facts not in context).",
        "fix": "Strengthen system prompt: 'Answer ONLY from context'. Set temperature=0. Add an 'I don't know' fallback.",
    },
    "hallucination_rate": {
        "label": "Hallucination Rate", "group": "Generation", "direction": "lower",
        "target": 0.10, "limit": 0.20,
        "asks": "What share of the answer's claims is NOT supported by the context?",
        "diagnosis": "The LLM invents facts beyond the retrieved context (alert if > 10%).",
        "fix": "'Answer ONLY from context', temperature=0, 'I don't know' fallback, force citations, "
               "add a post-generation faithfulness check, discard chunks below the similarity threshold.",
    },
    "answer_relevancy": {
        "label": "Answer Relevancy", "group": "Generation", "direction": "higher",
        "target": 0.85, "limit": 0.80,
        "asks": "Does the answer actually address the question?",
        "diagnosis": "Answers are OFF-TOPIC (not addressing what the user asked).",
        "fix": "Improve prompt template. Add: 'Answer the question directly and specifically.'",
    },
    "answer_correctness": {
        "label": "Answer Correctness", "group": "Generation", "direction": "higher",
        "target": 0.85, "limit": 0.80,
        "asks": "Is the answer factually correct vs ground truth?",
        "diagnosis": "Answer is WRONG compared to ground truth.",
        "fix": "Check retrieval first. If retrieval is good, fix the prompt or use a better LLM.",
    },
}


def status(metric, score):
    """Return 'pass' | 'warn' | 'fail' | 'na'."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "na"
    m = METRICS[metric]
    if m["direction"] == "higher":
        return "pass" if score >= m["target"] else "warn" if score >= m["limit"] else "fail"
    return "pass" if score <= m["target"] else "warn" if score <= m["limit"] else "fail"


STATUS_ICON = {"pass": "🟢 Pass", "warn": "🟡 Needs work", "fail": "🔴 Fix needed", "na": "⚪ n/a"}
