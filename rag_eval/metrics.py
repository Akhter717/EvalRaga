"""Metric registry. Each metric is a function  fn(judge, sample) -> dict  decorated with @metric(...outputs).

sample = {"question": str, "answer": str, "contexts": [str], "ground_truth": str}

To add your own metric:

    @metric("conciseness")
    def conciseness(judge, s):
        r = judge.ask(f"Rate how concise this answer is 0-1: {s['answer']} Return: {{\"score\":0.0}}")
        return {"conciseness": r["score"]}

(then add an entry for it in config.METRICS so it gets a target and a fix.)
Keys starting with "detail_" are kept in the results as extra text columns.
"""
import numpy as np

from .llm import embed

NAN = float("nan")
REGISTRY = {}  # function name -> (function, outputs)


def metric(*outputs):
    def deco(fn):
        REGISTRY[fn.__name__] = (fn, tuple(outputs))
        return fn
    return deco


def fmt_ctx(contexts):
    return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


@metric("faithfulness", "hallucination_rate")
def faithfulness_and_hallucination(judge, s):
    """One LLM call gives both: faithfulness = supported/total claims, hallucination_rate = 1 - faithfulness."""
    r = judge.ask(
        "Break the ANSWER into separate factual claims. For each claim, say whether it is directly stated or "
        "clearly implied by the CONTEXT. Facts that come from general knowledge but are NOT in the context "
        "count as NOT supported.\n"
        f"CONTEXT:\n{fmt_ctx(s['contexts'])}\n\nANSWER: {s['answer']}\n\n"
        'Return: {"claims":[{"claim":"...","supported":true}]}'
    )
    claims = r.get("claims", [])
    if not claims:
        return {"faithfulness": NAN, "hallucination_rate": NAN}
    supported = sum(bool(c.get("supported")) for c in claims)
    bad = [str(c.get("claim", "")) for c in claims if not c.get("supported")]
    f = supported / len(claims)
    return {"faithfulness": f, "hallucination_rate": 1 - f, "detail_unsupported_claims": "; ".join(bad)}


@metric("answer_relevancy")
def answer_relevancy(judge, s):
    r = judge.ask(
        "Generate 3 different questions that the following ANSWER would be a good response to.\n"
        f"ANSWER: {s['answer']}\n\n"
        'Return: {"questions":["...","...","..."]}'
    )
    qs = [x for x in r.get("questions", []) if isinstance(x, str) and x.strip()]
    if not qs:
        return {"answer_relevancy": NAN}
    v = embed([s["question"]] + qs)
    return {"answer_relevancy": float(np.clip(np.mean(v[1:] @ v[0]), 0, 1))}


@metric("context_precision")
def context_precision(judge, s):
    r = judge.ask(
        "For each numbered CHUNK, decide if it is relevant for answering the QUESTION "
        "(use the reference answer as a guide).\n"
        f"QUESTION: {s['question']}\nREFERENCE: {s['ground_truth']}\nCHUNKS:\n{fmt_ctx(s['contexts'])}\n\n"
        'Return: {"verdicts":[{"chunk":1,"relevant":true}]}'
    )
    v = r.get("verdicts", [])
    if not v or not s["contexts"]:
        return {"context_precision": NAN}
    return {"context_precision": sum(bool(x.get("relevant")) for x in v) / len(s["contexts"])}


@metric("context_recall")
def context_recall(judge, s):
    r = judge.ask(
        "Split the REFERENCE answer into individual statements. For each, say whether it can be "
        "found in (attributed to) the CONTEXT.\n"
        f"CONTEXT:\n{fmt_ctx(s['contexts'])}\n\nREFERENCE: {s['ground_truth']}\n\n"
        'Return: {"statements":[{"statement":"...","attributed":true}]}'
    )
    st_ = r.get("statements", [])
    if not st_:
        return {"context_recall": NAN}
    return {"context_recall": sum(bool(x.get("attributed")) for x in st_) / len(st_)}


@metric("answer_correctness")
def answer_correctness(judge, s):
    r = judge.ask(
        "Compare the ANSWER with the REFERENCE. List facts as:\n"
        "TP = facts in both, FP = facts in the answer but not in the reference (or wrong), "
        "FN = facts in the reference missing from the answer.\n"
        f"QUESTION: {s['question']}\nREFERENCE: {s['ground_truth']}\nANSWER: {s['answer']}\n\n"
        'Return: {"TP":["..."],"FP":["..."],"FN":["..."]}'
    )
    tp, fp, fn = len(r.get("TP", [])), len(r.get("FP", [])), len(r.get("FN", []))
    denom = tp + 0.5 * (fp + fn)
    f1 = tp / denom if denom else 0.0
    v = embed([s["answer"], s["ground_truth"]])
    sim = float(np.clip(v[0] @ v[1], 0, 1))
    return {"answer_correctness": 0.75 * f1 + 0.25 * sim}  # same 75/25 weighting RAGAS uses
