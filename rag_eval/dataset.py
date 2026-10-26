"""Dataset helpers: load a test file, parse contexts, chunk a document, build a dataset from a document.

A dataset is a table with 4 columns: question, answer, contexts, ground_truth.
No Streamlit dependency.
"""
import json
import os

import numpy as np
import pandas as pd

REQUIRED = ["question", "answer", "contexts", "ground_truth"]


def parse_contexts(value):
    """Turn a cell into a list of context strings.

    Accepts: a list, a JSON list string, or a string with chunks separated by '||'.
    """
    if isinstance(value, (list, tuple)):
        return [str(c) for c in value if str(c).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(c) for c in data if str(c).strip()]
        except json.JSONDecodeError:
            pass
    return [c.strip() for c in text.split("||") if c.strip()]


def load_dataset(file):
    """Read .csv / .json / .xlsx (a path or a Streamlit uploaded file) and check the columns."""
    name = getattr(file, "name", str(file)).lower()
    ext = os.path.splitext(name)[1]
    if ext == ".csv":
        df = pd.read_csv(file)
    elif ext == ".json":
        df = pd.read_json(file)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Use a .csv, .json or .xlsx file.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED)}")
    df = df[REQUIRED].dropna(subset=["question"]).reset_index(drop=True)
    df["contexts"] = df["contexts"].apply(parse_contexts)
    return df


def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into chunks of `chunk_size` words, each overlapping the previous one by `overlap` words."""
    words = str(text).split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)
            if words[i:i + chunk_size]]


def build_dataset(text, qa_pairs, judge, k=3, chunk_size=120, overlap=20):
    """Mini RAG pipeline so you can test a document without your own RAG system.

    1) chunk the document  2) embed chunks  3) for each question take the top-k chunks
    4) ask the LLM to answer using ONLY those chunks.
    qa_pairs = [(question, ground_truth), ...]
    """
    from .llm import embed
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        raise ValueError("The document is empty.")
    chunk_vecs = embed(chunks)
    rows = []
    for question, truth in qa_pairs:
        q_vec = embed([question])[0]
        top = np.argsort(-(chunk_vecs @ q_vec))[:k]
        ctx = [chunks[i] for i in top]
        numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctx))
        r = judge.ask(
            "Answer the QUESTION using ONLY the CONTEXT. If the answer is not in the context, "
            "say you don't know.\n"
            f"CONTEXT:\n{numbered}\n\nQUESTION: {question}\n\n"
            'Return: {"answer":"..."}'
        )
        rows.append({"question": question, "answer": str(r.get("answer", "")),
                     "contexts": ctx, "ground_truth": truth})
    return pd.DataFrame(rows)


DEMO = pd.DataFrame([
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "contexts": ["Paris is the capital and largest city of France.",
                     "France is a country in Western Europe."],
        "ground_truth": "Paris is the capital of France.",
    },
    {
        "question": "When was the Eiffel Tower completed?",
        "answer": "The Eiffel Tower was completed in 1889 and is made of iron. It was designed by Leonardo da Vinci.",
        "contexts": ["The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
                     "The tower is made of wrought iron and is 330 metres tall."],
        "ground_truth": "The Eiffel Tower was completed in 1889.",
    },
    {
        "question": "What does RAG stand for?",
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "contexts": ["Python is a popular programming language.",
                     "Retrieval-Augmented Generation (RAG) combines a retriever with a language model."],
        "ground_truth": "RAG stands for Retrieval-Augmented Generation.",
    },
])
