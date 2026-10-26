"""RAG Evaluation Framework - Streamlit app (entry point for Streamlit Cloud)."""
import pandas as pd
import streamlit as st

from rag_eval import (METRICS, evaluate, summarize, resolve_functions,
                      load_dataset, build_dataset, DEMO)
from rag_eval.config import STATUS_ICON
from rag_eval.dataset import parse_contexts, extract_pdf_text
from rag_eval.llm import GroqJudge, MODELS

# These metrics compare against the correct answer, so they need a ground truth.
NEEDS_TRUTH = {"context_precision", "context_recall", "answer_correctness"}

st.set_page_config(page_title="RAG Evaluation Framework", page_icon="🧪", layout="wide")
st.title("🧪 RAG Evaluation Framework")
st.caption("Reusable RAGAS-style evaluation: retrieval quality, faithfulness, relevance and hallucination rate.")

ss = st.session_state
for key in ("df", "results", "summary", "errors"):
    ss.setdefault(key, None)
ss.setdefault("quick_rows", [])
ss.setdefault("quick_msg", "")
ss.setdefault("doc_text", "")
for key in ("q_question", "q_answer", "q_contexts", "q_truth"):
    ss.setdefault(key, "")


def secret_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = secret_key()
    if api_key:
        st.success("Groq API key loaded from Streamlit secrets.")
    else:
        api_key = st.text_input("Groq API key", type="password", help="Free key from console.groq.com")
    model_label = st.selectbox("Judge model", list(MODELS), index=0)
    delay = st.slider("Delay between LLM calls (sec)", 0.0, 6.0, 2.0, 0.5,
                      help="Raise this if you hit Groq free-tier rate limits.")
    st.divider()
    st.markdown("**Metrics available**")
    for group in ("Retrieval", "Generation"):
        names = [m["label"] for m in METRICS.values() if m["group"] == group]
        st.markdown(f"*{group}:* " + ", ".join(names))


# ---------------------------------------------------------------- helpers
def make_judge():
    if not api_key:
        st.error("Add your Groq API key in the sidebar (or in Streamlit secrets) first.")
        st.stop()
    return GroqJudge(api_key, MODELS[model_label], delay)


def show_df(df):
    view = df.copy()
    view["contexts"] = view["contexts"].apply(lambda c: " || ".join(c) if isinstance(c, list) else c)
    st.dataframe(view, use_container_width=True, hide_index=True)


def available_metrics(df):
    """If any row has no ground truth, drop the metrics that need one."""
    blank = df["ground_truth"].astype(str).str.strip().eq("").any()
    return [m for m in METRICS if not (blank and m in NEEDS_TRUTH)]


def run_eval(selected):
    judge = make_judge()
    bar = st.progress(0.0, text="Starting...")
    res, errs = evaluate(ss.df, judge, selected,
                         progress=lambda frac, text: bar.progress(min(frac, 1.0), text=text))
    ss.results, ss.errors, ss.summary = res, errs, summarize(res)
    bar.empty()


def split_chunks(text):
    """Chunks separated by '||' or by new lines."""
    if "||" in text:
        return parse_contexts(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_current():
    """Read the Quick test form. Returns (row, error)."""
    q, a = ss.q_question.strip(), ss.q_answer.strip()
    chunks, truth = split_chunks(ss.q_contexts), ss.q_truth.strip()
    if not (q or a or chunks or truth):
        return None, None
    if not (q and a and chunks):
        return None, "Question, Answer and Chunks - all three are required."
    return {"question": q, "answer": a, "contexts": chunks, "ground_truth": truth}, None


def add_quick():
    ss.quick_msg = ""
    row, err = read_current()
    if err:
        ss.quick_msg = err
    elif row is None:
        ss.quick_msg = "Form is empty."
    else:
        ss.quick_rows.append(row)
        for key in ("q_question", "q_answer", "q_contexts", "q_truth"):
            ss[key] = ""


def clear_quick():
    ss.quick_rows = []
    ss.quick_msg = ""


# ---------------------------------------------------------------- step 1: dataset
st.header("1️⃣ Test scenario / dataset")
QUICK, DOC, UPLOAD, DEMO_OPT = ("Quick test (type a scenario)", "Document / PDF (build RAG)",
                                "Upload a file", "Demo data")
source = st.radio("How do you want to give the data?", [QUICK, DOC, UPLOAD, DEMO_OPT])

if source == QUICK:
    st.caption("Type a scenario given to you live: question, the answer the RAG gave, and the chunks it used. "
               "Ground truth is optional - without it, only Faithfulness, Hallucination Rate and "
               "Answer Relevancy are scored.")
    st.text_area("Question", key="q_question", height=70)
    st.text_area("Answer (what the RAG replied)", key="q_answer", height=100)
    st.text_area("Chunks / context (one chunk per line, or separate with ||)", key="q_contexts", height=140)
    st.text_input("Ground truth - the correct answer (optional)", key="q_truth")
    c1, c2, c3 = st.columns(3)
    c1.button("➕ Add to list", on_click=add_quick, use_container_width=True)
    run_now = c2.button("▶️ Test now", type="primary", use_container_width=True)
    c3.button("🗑️ Clear list", on_click=clear_quick, use_container_width=True)
    if ss.quick_msg:
        st.warning(ss.quick_msg)
    if ss.quick_rows:
        st.caption(f"Scenarios in the list: {len(ss.quick_rows)}")
        show_df(pd.DataFrame(ss.quick_rows))
    if run_now:
        cur, err = read_current()
        if err:
            st.error(err)
        else:
            rows = list(ss.quick_rows) + ([cur] if cur else [])
            if not rows:
                st.error("Fill the form first.")
            else:
                ss.df, ss.results, ss.summary = pd.DataFrame(rows), None, None
                run_eval(available_metrics(ss.df))

elif source == DOC:
    st.caption("Upload a text PDF (or paste text). The app chunks it, retrieves the best chunks for each "
               "question and generates the answers, then the judge scores them.")
    pdf = st.file_uploader("PDF (optional)", type=["pdf"])
    pages = st.number_input("Max pages to read", 1, 200, 10)
    if pdf is not None:
        sig = (pdf.name, pdf.size, int(pages))
        if ss.get("pdf_sig") != sig:
            try:
                ss.doc_text = extract_pdf_text(pdf, int(pages))
                ss.pdf_sig = sig
            except Exception as e:
                st.error(f"Could not read the PDF: {e}")
    doc = st.text_area("Document text (from the PDF, or paste your own)", key="doc_text", height=180)
    if doc.strip():
        st.caption(f"{len(doc.split())} words")
    st.caption("Add questions. Ground truth is optional (without it, 3 of the 6 metrics are skipped).")
    qa = st.data_editor(pd.DataFrame({"question": [""], "ground_truth": [""]}),
                        num_rows="dynamic", use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    k = c1.slider("Chunks retrieved per question (k)", 1, 7, 3)
    size = c2.slider("Chunk size (words)", 40, 300, 120, 10)
    if st.button("Build dataset"):
        pairs = [(str(r.question).strip(), str(r.ground_truth or "").strip())
                 for r in qa.itertuples() if str(r.question).strip()]
        if not doc.strip() or not pairs:
            st.error("Add a document (or PDF) and at least one question.")
        else:
            with st.spinner("Retrieving chunks and generating answers..."):
                try:
                    ss.df = build_dataset(doc, pairs, make_judge(), k=k, chunk_size=size, overlap=size // 6)
                    ss.results, ss.summary = None, None
                except Exception as e:
                    st.error(f"Could not build dataset: {e}")

elif source == UPLOAD:
    st.caption("CSV / JSON / XLSX with columns: question, answer, contexts, ground_truth. "
               "In a CSV, separate multiple contexts with ||")
    up = st.file_uploader("Dataset file", type=["csv", "json", "xlsx"])
    if up is not None:
        try:
            ss.df, ss.results, ss.summary = load_dataset(up), None, None
        except Exception as e:
            st.error(str(e))

else:
    if st.button("Load demo data"):
        ss.df, ss.results, ss.summary = DEMO.copy(), None, None

if ss.df is not None:
    st.success(f"Dataset ready: {len(ss.df)} rows")
    show_df(ss.df)

# ---------------------------------------------------------------- step 2: run
if ss.df is not None:
    st.header("2️⃣ Run evaluation")
    avail = available_metrics(ss.df)
    if len(avail) < len(METRICS):
        st.info("Some rows have no ground truth, so Context Precision, Context Recall and "
                "Answer Correctness are skipped.")
    selected = st.multiselect("Metrics to run", avail, default=avail,
                              format_func=lambda m: f"{METRICS[m]['label']} ({METRICS[m]['group']})")
    calls = len(ss.df) * len(resolve_functions(selected))
    st.caption(f"About {calls} LLM calls (~{int(calls * (delay + 2))} sec).")

    if st.button("▶️ Run evaluation", type="primary", disabled=not selected):
        run_eval(selected)

# ---------------------------------------------------------------- step 3: results
if ss.results is not None and ss.summary is not None:
    st.header("3️⃣ Results")
    summ = ss.summary

    if summ.get("_production_ready"):
        st.success("✅ All selected metrics pass - production ready.")
    else:
        st.warning("⚠️ Some metrics need work - see the fixes below.")

    rows = []
    for m, v in summ.items():
        if m.startswith("_"):
            continue
        cfg = METRICS[m]
        sign = "≥" if cfg["direction"] == "higher" else "≤"
        rows.append({"Metric": cfg["label"], "Group": cfg["group"],
                     "Score": None if pd.isna(v["score"]) else round(v["score"], 3),
                     "Target": f"{sign} {cfg['target']:.2f}", "Status": STATUS_ICON[v["status"]]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if "_hallucinated_answers" in summ:
        h = summ["_hallucinated_answers"]
        st.metric("Answers with at least one hallucination", f"{h['count']} / {h['total']}")

    bad = [m for m, v in summ.items() if not m.startswith("_") and v["status"] in ("warn", "fail")]
    if bad:
        st.subheader("🔧 What to fix")
        for m in bad:
            cfg = METRICS[m]
            with st.expander(f"{cfg['label']} - {STATUS_ICON[summ[m]['status']]}"):
                st.markdown(f"**Question it answers:** {cfg['asks']}")
                st.markdown(f"**Diagnosis:** {cfg['diagnosis']}")
                st.markdown(f"**Fix:** {cfg['fix']}")

    st.subheader("Per-question results")
    st.dataframe(ss.results, use_container_width=True, hide_index=True)

    if ss.errors:
        with st.expander(f"⚠️ {len(ss.errors)} error(s) during the run"):
            for e in ss.errors:
                st.text(e)

    st.download_button("⬇️ Download results (CSV)", ss.results.to_csv(index=False).encode("utf-8"),
                       file_name="rag_eval_results.csv", mime="text/csv")
