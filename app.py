import json
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
intent_descriptions = config["intent_descriptions"]

st.title("🏦 Multi-Paradigm Support Ticket Triage Router")
st.markdown(
    "Built from **Hands-On Large Language Models (Chapter 4)** on the **Banking77** dataset. "
    "Compare a **Supervised Classifier (Frozen BGE Embeddings + Logistic Regression)** against "
    "**Zero-Shot Label-Description Cosine Similarity** in real time."
)

tab_demo, tab_bench = st.tabs(["🚀 Live Ticket Router", "📊 Chapter 4 Benchmark Results"])

with tab_demo:
    col_input, col_settings = st.columns([2, 1])

    with col_settings:
        threshold = st.slider(
            "Auto-Routing Confidence Threshold",
            min_value=0.30,
            max_value=0.95,
            value=0.65,
            step=0.05,
            help="Tickets below this probability escalate to human review or a slower generative LLM."
        )
        preset = st.selectbox(
            "Or pick a sample customer ticket:",
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
            placeholder="Describe a banking issue (e.g., lost card, duplicate charge, pending payment)..."
        )
        analyze_clicked = st.button("Classify & Route Ticket", type="primary", use_container_width=True)

    if analyze_clicked and ticket_text.strip():
        vec = embed_model.encode([ticket_text], normalize_embeddings=True)

        # 1. Supervised Logistic Regression probabilities
        sup_probs = clf.predict_proba(vec)[0]
        top_sup_idx = int(np.argmax(sup_probs))
        top_sup_conf = float(sup_probs[top_sup_idx])
        top_sup_label = clean_names[top_sup_idx]

        # 2. Zero-Shot Cosine Similarity scores
        cos_sims = np.matmul(vec, desc_vecs.T)[0]
        top_zs_idx = int(np.argmax(cos_sims))
        top_zs_label = clean_names[top_zs_idx]

        st.divider()
        if top_sup_conf >= threshold:
            st.success(
                f"**AUTO-ROUTED** → Queue: **`{top_sup_label}`** | "
                f"Confidence: **{top_sup_conf:.1%}** (Threshold: {threshold:.0%})"
            )
        else:
            st.warning(
                f"**ESCALATE TO HUMAN / LLM FALLBACK** → Top supervised prediction **`{top_sup_label}`** "
                f"({top_sup_conf:.1%}) is below the {threshold:.0%} confidence threshold."
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Supervised (Frozen BGE + LogReg)")
            sup_df = pd.DataFrame({
                "Intent": clean_names,
                "Probability": sup_probs
            }).sort_values("Probability", ascending=False).head(5)
            st.bar_chart(sup_df.set_index("Intent"))

        with c2:
            st.subheader("Zero-Shot Cosine Similarity")
            st.caption(f"Top Zero-Shot Match: `{top_zs_label}` (Cosine Sim: {cos_sims[top_zs_idx]:.3f})")
            zs_df = pd.DataFrame({
                "Intent": clean_names,
                "Cosine Similarity": cos_sims
            }).sort_values("Cosine Similarity", ascending=False).head(5)
            st.bar_chart(zs_df.set_index("Intent"))

    elif analyze_clicked:
        st.error("Please enter a ticket message first.")

with tab_bench:
    st.subheader("4-Paradigm Evaluation on Banking77 (10 Confusable Intents, 400 Test Tickets)")
    bench_df = pd.DataFrame(config["benchmark_summary"])
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Key Engineering Takeaways
    * **Prompting / Description Quality Matters:** Replacing 3-word labels in Zero-Shot NLI (`0.541` F1) with single-sentence semantic descriptions in Zero-Shot Embedding Similarity (`0.864` F1) boosted Macro F1 by **+32.3%** with zero labeled examples while running **50× faster**.
    * **Data Efficiency of Frozen Embeddings:** With just **10 labeled examples per class**, `LogisticRegression` on `BAAI/bge-small-en-v1.5` surpassed all zero-shot baselines (`~0.918` F1), reaching **`0.970` F1** on the full training split at **1,555+ tickets/sec**.
    * **Confidence-Based Cascading:** Using calibrated probabilities from the lightweight embedding classifier lets us auto-route high-confidence tickets in milliseconds and reserve expensive generative models or human agents strictly for ambiguous edge cases.
    """)