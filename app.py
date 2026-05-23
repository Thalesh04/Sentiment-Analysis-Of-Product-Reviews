"""
SentiScope — Streamlit entry point.

The model trains once at startup from the CSV specified in config.py.
Visitors see a statistics dashboard and can analyse any review live.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

import config
from src.preprocessing.cleaner import clean_text
from src.preprocessing.dataset import load_and_prepare
from src.models.trainer import train_models
from src.ui.styles import inject_css
from src.ui.components import (
    render_hero,
    render_dataset_stats,
    render_metrics_cards,
    render_analyser,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# ── Startup: load data + train (cached — runs once per server process) ─────────
@st.cache_resource(show_spinner=False)
def startup():
    """Load CSV, compute stats, train models. Cached for the lifetime of the server."""
    if not config.DATA_PATH.exists():
        return None, None, None

    df, stats = load_and_prepare(
        config.DATA_PATH,
        config.SAMPLE_SIZE,
        config.RANDOM_STATE,
    )
    bundle = train_models(df)
    return bundle, stats, True


with st.spinner("⚙️ Loading dataset and training models — just a moment..."):
    bundle, stats, ok = startup()

# ── Guard: missing CSV ─────────────────────────────────────────────────────────
if not ok:
    render_hero()
    st.error(
        f"**Dataset not found:** `{config.DATA_PATH}`\n\n"
        "Place your `Reviews.csv` in the `data/` folder (or update `DATA_PATH` in `config.py`), "
        "then restart the server."
    )
    st.stop()

# ── Page ───────────────────────────────────────────────────────────────────────
render_hero()
render_dataset_stats(stats)

st.markdown("---")
render_metrics_cards(bundle.lr_metrics, bundle.nb_metrics)

st.markdown("---")
render_analyser(bundle, "Logistic Regression", clean_text)
