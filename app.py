import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from digital_scientist.pipeline import run as run_pipeline

load_dotenv()

st.set_page_config(
    page_title="AI Drug Discovery",
    page_icon="🧬",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
    .subtitle   { color: #555; margin-bottom: 2rem; }
    .step-badge {
        background: #e8f4fd; color: #1565c0;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
        display: inline-block; margin-bottom: 8px;
    }
    .summary-box {
        background: #f8f9fa; border-left: 4px solid #1565c0;
        padding: 1.2rem 1.5rem; border-radius: 4px;
        line-height: 1.7;
    }
    .metric-card {
        background: #fff; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 AI Drug Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-agent pipeline · Open Targets · ChEMBL · Neural Network · Gemini</div>', unsafe_allow_html=True)

# ── Input ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    disease = st.text_input("Disease", placeholder="Enter a disease (e.g. lung cancer, diabetes, Alzheimer's disease)", label_visibility="collapsed")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Discover Drugs", type="primary", use_container_width=True)

st.markdown("**Try:** lung cancer · heart disease · diabetes · Alzheimer's disease · asthma")

# ── Pipeline ────────────────────────────────────────────────────────────────
if run and disease.strip():
    with st.spinner("Running multi-agent pipeline..."):
        progress = st.empty()
        progress.info("Step 1/3 · Biology Agent — finding targets...")
        try:
            result = run_pipeline(disease.strip())
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        progress.empty()

    targets    = result.get("targets", [])
    candidates = result.get("candidates", [])

    # ── Metrics ─────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Disease", result.get("disease", disease))
    m2.metric("Targets Found", len(targets))
    m3.metric("Drug-Like Candidates", len(candidates))
    top_score = max((c.get("activity_score", 0) for c in candidates), default=0)
    m4.metric("Top AI Bioactivity Score", f"{top_score:.2%}")

    lookup = result.get("disease_lookup", {})
    resolver = lookup.get("resolver", "direct")
    query_used = lookup.get("query_used", disease)
    resolver_label = {
        "direct": "Direct",
        "alias": "Local Alias",
        "fuzzy": "Local Fuzzy",
        "grok": "Grok Fallback",
    }.get(resolver, resolver.title())
    st.caption(f"Disease resolver: {resolver_label} | Query used: {query_used}")

    # ── Top model pick ───────────────────────────────────────
    if candidates and candidates[0].get("activity_score"):
        top = candidates[0]
        st.markdown("---")
        st.markdown('<div class="step-badge">Neural Network · Top Pick</div>', unsafe_allow_html=True)
        st.subheader(f"🧠 {top.get('name', top.get('chembl_id'))} — AI Score: {top['activity_score']:.2%}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target", top.get("target_symbol", "?"))
        c2.metric("Molecular Weight", f"{top.get('mw', '?')} Da")
        c3.metric("LogP", top.get("logp", "?"))
        c4.metric("pChEMBL", top.get("pchembl", "?"))
        st.progress(float(top["activity_score"]), text=f"Predicted bioactivity: {top['activity_score']:.2%}")

    st.markdown("---")
    left, right = st.columns(2)

    # ── Targets table ────────────────────────────────────────
    with left:
        st.markdown('<div class="step-badge">Step 1 · Biology Agent</div>', unsafe_allow_html=True)
        st.subheader("Top Biological Targets")
        if targets:
            df_targets = pd.DataFrame(targets)[["symbol", "name", "score", "ensembl_id"]]
            df_targets.columns = ["Symbol", "Name", "OT Score", "Ensembl ID"]
            df_targets["OT Score"] = df_targets["OT Score"].apply(lambda x: f"{x:.4f}")
            st.dataframe(df_targets, use_container_width=True, hide_index=True)
        else:
            st.warning("No targets found.")

    # ── Candidates table ─────────────────────────────────────
    with right:
        st.markdown('<div class="step-badge">Step 2 · Chemistry Agent</div>', unsafe_allow_html=True)
        st.subheader("Drug-Like Candidates")
        if candidates:
            cols = ["name", "target_symbol", "mw", "logp", "pchembl", "activity_score"]
            available = [c for c in cols if c in candidates[0]]
            df_cands = pd.DataFrame(candidates[:15])[available]
            df_cands.columns = [c.replace("_", " ").title() for c in available]
            if "Activity Score" in df_cands.columns:
                df_cands["Activity Score"] = df_cands["Activity Score"].apply(
                    lambda x: f"{float(x):.4f}" if x else "N/A"
                )
            st.dataframe(df_cands, use_container_width=True, hide_index=True)
        else:
            st.warning("No drug-like candidates found.")

    # ── AI Summaries ─────────────────────────────────────────
    st.markdown("---")
    col_bio, col_chem = st.columns(2)

    with col_bio:
        st.markdown('<div class="step-badge">Biology Agent</div>', unsafe_allow_html=True)
        st.subheader("Target Analysis")
        st.markdown(f'<div class="summary-box">{result.get("biology_summary","")}</div>', unsafe_allow_html=True)

    with col_chem:
        st.markdown('<div class="step-badge">Chemistry Agent</div>', unsafe_allow_html=True)
        st.subheader("Candidate Analysis")
        st.markdown(f'<div class="summary-box">{result.get("chemistry_summary","")}</div>', unsafe_allow_html=True)

elif run and not disease.strip():
    st.warning("Please enter a disease name.")
