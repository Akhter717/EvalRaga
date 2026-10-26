"""rag_eval - a small, reusable RAG testing framework (RAGAS-style metrics + hallucination rate)."""
from .config import METRICS, status
from .runner import evaluate, summarize, resolve_functions
from .dataset import load_dataset, parse_contexts, chunk_text, build_dataset, DEMO

__all__ = ["METRICS", "status", "evaluate", "summarize", "resolve_functions",
           "load_dataset", "parse_contexts", "chunk_text", "build_dataset", "DEMO"]
