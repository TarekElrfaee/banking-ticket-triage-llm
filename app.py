import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Banking Ticket Triage Router",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    with open("router_config.json", "r") as f:
        config = json.load(f)
    clf = joblib.load("ticket_router_clf.joblib")
    embed_model = SentenceTransformer(config["embedding_model"], device="cpu")
    desc_vecs = embed_model.encode(config["intent_descriptions"], normalize_embeddings=True)
    return config, clf, embed_model, desc_vecs

config, clf, embed_model, desc_vecs = load_pipeline()
clean_names = config["clean_names"]

st.title("🏦 Multi-Paradigm Support Ticket Triage Router")
st.caption(
    "End-to-end NLP classification & confidence-based routing pipeline built on Banking77 "
    "(DeBERTa-v3 NLI vs. Zero-Shot Embeddings vs. Frozen BGE + Logistic Regression vs. FLAN-T5)."
)

tab_single, tab_batch, tab_eval = st.tabs([
    "🎯 Single Ticket Triage",
    "📂 Batch Queue Processing",
    "📊 Benchmark & Error Analysis"
])

# ==========================================
# TAB 1: SINGLE TICKET TRIAGE
# ==========================================
with tab_single:
    col_input, col_settings = st.columns([2, 1])

    with col_settings:
        threshold = st.slider(
            "Auto-Routing Confidence Threshold",
            min_value=0.30,
            max_value=0.95,
            value=0.65,
            step=0.05
        )
        preset = st.selectbox(
            "Load sample edge-case ticket:",
            [
                "-- Custom input --",
                "I tried paying at the grocery store twice and both transactions went through!",
                "My new card was supposed to arrive Tuesday, can you check where it is?",
                "The merchant cancelled my order last week, why is the charge still pending?",
                "Someone took my wallet on the subway, please freeze my account right now.",
                "Why did my payment get reversed automatically at checkout?"
            ]
        )

    with col_input:
        default_text = "" if preset == "-- Custom input --" else preset
        ticket_text = st.text_area(
            "Incoming Customer Support Ticket",
            value=default_text,
            height=130,
            placeholder="Enter a customer banking inquiry..."
        )
        analyze_clicked = st.button("Classify & Route Ticket", type="primary", width="stretch")

    if analyze_clicked and ticket_text.strip():
        t0 = time.perf_counter()
        vec = embed_model.encode([ticket_text], normalize_embeddings=True)
        sup_probs = clf.predict_proba(vec)[0]
        cos_sims = np.matmul(vec, desc_vecs.T)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        top_sup_idx = int(np.argmax(sup_probs))
        top_sup_conf = float(sup_probs[top_sup_idx])
        top_sup_label = clean_names[top_sup_idx]
        top_zs_idx = int(np.argmax(cos_sims))

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Queue", top_sup_label)
        m2.metric("Model Confidence", f"{top_sup_conf:.1%}")
        m3.metric("CPU Inference Latency", f"{latency_ms:.1f} ms")

        if top_sup_conf >= threshold:
            st.success(f"**AUTO-ROUTED** to `{top_sup_label}` (exceeds {threshold:.0%} threshold).")
        else:
            st.warning(f"**ESCALATED TO HUMAN / LLM** (`{top_sup_conf:.1%}` is below {threshold:.0%} threshold).")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Supervised (Frozen BGE + LogReg)")
            sup_df = pd.DataFrame({"Intent": clean_names, "Probability": sup_probs}).sort_values("Probability", ascending=False).head(5)
            st.bar_chart(sup_df.set_index("Intent"))

        with c2:
            st.subheader("Zero-Shot Cosine Similarity")
            st.caption(f"Top Zero-Shot Match: `{clean_names[top_zs_idx]}` ({cos_sims[top_zs_idx]:.3f})")
            zs_df = pd.DataFrame({"Intent": clean_names, "Cosine Similarity": cos_sims}).sort_values("Cosine Similarity", ascending=False).head(5)
            st.bar_chart(zs_df.set_index("Intent"))

# ==========================================
# TAB 2: BATCH CSV TRIAGE
# ==========================================
with tab_batch:
    st.subheader("Batch Support Queue Routing")
    st.write("Upload a CSV with a `text` column—or run the built-in 25-ticket sample batch.")

    batch_thresh = st.slider("Batch Confidence Threshold", 0.30, 0.95, 0.70, 0.05, key="batch_thresh")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    batch_texts = None
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        if "text" in raw_df.columns:
            batch_texts = raw_df["text"].dropna().astype(str).tolist()
        else:
            st.error("CSV must contain a column named `text`.")
    elif st.button("Run Demo on 25 Unseen Test Tickets", type="primary"):
        if "test_eval_records" in config:
            batch_texts = [r["text"] for r in config["test_eval_records"][:25]]

    if batch_texts:
        t0 = time.perf_counter()
        vecs = embed_model.encode(batch_texts, batch_size=32, normalize_embeddings=True)
        probs = clf.predict_proba(vecs)
        total_ms = (time.perf_counter() - t0) * 1000

        pred_ids = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        actions = ["Auto-Route" if c >= batch_thresh else "Escalate" for c in confs]

        out_df = pd.DataFrame({
            "Ticket": batch_texts,
            "Routed_Intent": [clean_names[i] for i in pred_ids],
            "Confidence": np.round(confs, 4),
            "Routing_Action": actions
        })

        k1, k2, k3 = st.columns(3)
        k1.metric("Tickets Processed", len(out_df))
        k2.metric("Auto-Routed Rate", f"{(actions.count('Auto-Route') / len(out_df)):.1%}")
        k3.metric("Total Batch Time", f"{total_ms:.1f} ms")

        st.dataframe(out_df, width="stretch", hide_index=True)
        st.download_button(
            "Download Routed CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="routed_tickets.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 3: BENCHMARK & ERROR ANALYSIS
# ==========================================
with tab_eval:
    st.subheader("4-Paradigm Model Benchmark (Banking77 — 400 Test Tickets)")
    bench_df = pd.DataFrame(config["benchmark_summary"])
    st.dataframe(bench_df, width="stretch", hide_index=True)

    if "test_eval_records" in config:
        st.divider()
        st.subheader("Interactive Confidence Threshold Simulator (Test Set)")
        eval_df = pd.DataFrame(config["test_eval_records"])
        sim_thresh = st.slider("Simulate Production Confidence Threshold", 0.30, 0.98, 0.70, 0.02, key="sim_thresh")

        auto_df = eval_df[eval_df["confidence"] >= sim_thresh]
        coverage = len(auto_df) / len(eval_df)
        auto_acc = (auto_df["true_label"] == auto_df["pred_label"]).mean() if len(auto_df) > 0 else 0.0

        s1, s2, s3 = st.columns(3)
        s1.metric("Automation Coverage", f"{coverage:.1%}", f"{len(auto_df)} / {len(eval_df)} tickets")
        s2.metric("Auto-Routed Accuracy", f"{auto_acc:.1%}", f"+{(auto_acc - 0.970):.1%} vs uncalibrated")
        s3.metric("Human Escalation Queue", f"{1 - coverage:.1%}", f"{len(eval_df) - len(auto_df)} tickets")

        st.subheader("Misclassified Test Cases (Error Audit)")
        errors_df = eval_df[eval_df["true_label"] != eval_df["pred_label"]].sort_values("confidence", ascending=False)
        st.caption(f"Showing all {len(errors_df)} errors out of 400 test tickets:")
        st.dataframe(errors_df, width="stretch", hide_index=True)