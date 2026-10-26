"""Runs the selected metrics over a dataset and summarises the results."""
import numpy as np
import pandas as pd

from .config import METRICS, status
from .metrics import REGISTRY, NAN


def resolve_functions(selected):
    """Which metric functions must run for the selected output metrics (each function runs once)."""
    return [name for name, (_, outs) in REGISTRY.items() if set(outs) & set(selected)]


def evaluate(df, judge, selected, progress=None):
    """df needs columns: question, answer, contexts (list or str), ground_truth.
    progress(done, total, text) is optional. Returns (results_df, errors)."""
    from .dataset import parse_contexts
    fn_names = resolve_functions(selected)
    total, done = len(df) * len(fn_names), 0
    rows, errors = [], []
    for i, rec in enumerate(df.reset_index(drop=True).to_dict("records")):
        sample = {
            "question": str(rec["question"]), "answer": str(rec["answer"]),
            "contexts": parse_contexts(rec["contexts"]), "ground_truth": str(rec["ground_truth"]),
        }
        out = {"question": sample["question"], "answer": sample["answer"]}
        for name in fn_names:
            fn, outs = REGISTRY[name]
            if progress:
                progress(done / max(total, 1), f"Row {i + 1}/{len(df)} - {name}")
            try:
                out.update(fn(judge, sample))
            except Exception as e:  # keep going, report at the end
                out.update({o: NAN for o in outs})
                errors.append(f"Row {i + 1} / {name}: {e}")
            done += 1
        rows.append(out)
    if progress:
        progress(1.0, "Done")
    res = pd.DataFrame(rows)
    keep = ["question", "answer"] + [m for m in METRICS if m in selected and m in res.columns] \
        + [c for c in res.columns if c.startswith("detail_")]
    return res[keep], errors


def summarize(results):
    """Average score + status per metric, plus % of answers that contain at least one hallucination."""
    summary = {}
    for m in METRICS:
        if m in results.columns:
            avg = float(results[m].mean())
            summary[m] = {"score": avg, "status": status(m, avg)}
    if "hallucination_rate" in results.columns:
        col = results["hallucination_rate"].dropna()
        summary["_hallucinated_answers"] = {"count": int((col > 0).sum()), "total": int(len(col))}
    summary["_production_ready"] = bool(summary) and all(
        v["status"] == "pass" for k, v in summary.items() if not k.startswith("_"))
    return summary
