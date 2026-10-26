"""RAG Evaluation Framework - Streamlit app (entry point for Streamlit Cloud)."""
import pandas as pd
import streamlit as st

from rag_eval import (METRICS, evaluate, summarize, resolve_functions,
                      load_dataset, build_dataset, DEMO)
from rag_eval.config import STATUS_ICON
from rag_eval.llm import GroqJudge, MODELS, DEFAULT_MODEL

st.set_page_config(page_title="RAG Evaluation Framework", page_icon="🧪", layout="wide")
st.title("🧪 RAG Evaluation Framework")
st.caption("Reusable RAGAS-style evaluation: retrieval quality, faithfulness, relevance and hallucination rate.")

ss = st.session_state
for key in ("df", "results", "summary", "errors"):
    ss.setdefault(key, None)


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


def make_judge():
    if not api_key:
        st.error("Add your Groq API key in the sidebar (or in Streamlit secrets) first.")
        st.stop()
    return GroqJudge(api_key, MODELS[model_label], delay)


def show_df(df):
    view = df.copy()
    view["contexts"] = view["contexts"].apply(lambda c: " || ".join(c) if isinstance(c, list) else c)
    st.dataframe(view, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- step 1: dataset
st.header("1️⃣ Test dataset")
source = st.radio("Where does your data come from?",
                  ["Demo data", "Upload a file", "Build from a document"], horizontal=True)

if source == "Demo data":
    if st.button("Load demo data"):
        ss.df, ss.results, ss.summary = DEMO.copy(), None, None

elif source == "Upload a file":
    st.caption("CSV / JSON / XLSX with columns: question, answer, contexts, ground_truth. "
               "In a CSV, separate multiple contexts with ||")
    up = st.file_uploader("Dataset file", type=["csv", "json", "xlsx"])
    if up is not None:
        try:
            ss.df, ss.results, ss.summary = load_dataset(up), None, None
        except Exception as e:
            st.error(str(e))

else:
    doc = st.text_area("Paste your document text", height=180)
    st.caption("Add question + ground truth pairs. The app retrieves chunks and generates the answers for you.")
    qa = st.data_editor(pd.DataFrame({"question": [""], "ground_truth": [""]}),
                        num_rows="dynamic", use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    k = c1.slider("Chunks retrieved per question (k)", 1, 7, 3)
    size = c2.slider("Chunk size (words)", 40, 300, 120, 10)
    if st.button("Build dataset"):
        pairs = [(str(r.question).strip(), str(r.ground_truth).strip())
                 for r in qa.itertuples() if str(r.question).strip()]
        if not doc.strip() or not pairs:
            st.error("Add a document and at least one question.")
        else:
            with st.spinner("Retrieving chunks and generating answers..."):
                try:
                    ss.df = build_dataset(doc, pairs, make_judge(), k=k, chunk_size=size, overlap=size // 6)
                    ss.results, ss.summary = None, None
                except Exception as e:
                    st.error(f"Could not build dataset: {e}")

if ss.df is not None:
    st.success(f"Dataset ready: {len(ss.df)} rows")
    show_df(ss.df)

# ---------------------------------------------------------------- step 2: run
if ss.df is not None:
    st.header("2️⃣ Run evaluation")
    selected = st.multiselect("Metrics to run", list(METRICS), default=list(METRICS),
                              format_func=lambda m: f"{METRICS[m]['label']} ({METRICS[m]['group']})")
    calls = len(ss.df) * len(resolve_functions(selected))
    st.caption(f"About {calls} LLM calls (~{int(calls * (delay + 2))} sec).")

    if st.button("▶️ Run evaluation", type="primary", disabled=not selected):
        judge = make_judge()
        bar = st.progress(0.0, text="Starting...")
        res, errs = evaluate(ss.df, judge, selected,
                             progress=lambda frac, text: bar.progress(min(frac, 1.0), text=text))
        ss.results, ss.errors, ss.summary = res, errs, summarize(res)
        bar.empty()

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
