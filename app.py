# app.py

import streamlit as st
import pandas as pd
import os
import logging

from orchestrator.pipeline_orchestrator import PipelineOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit App Config ---
st.set_page_config(page_title="Multi-Agent Data Analyst", layout="wide")

# --- App Title ---
st.title("🤖 Multi-Agent Data Analyst")
st.markdown("""
This app uses a **Multi-Agent System (MAS)** powered by **local LLaMA / Mistral LLM agents** 
to analyze your data and generate intelligent reports. 🚀
""")

# --- File Upload ---
st.sidebar.header("1️⃣ Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Placeholder for dataframe + target column
df = None
target_column = None

if uploaded_file is not None:
    # Save uploaded file to docs/
    os.makedirs("docs", exist_ok=True)
    csv_path = os.path.join("docs", uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dataframe
    df = pd.read_csv(csv_path)
    st.success(f"✅ CSV loaded: {uploaded_file.name}")

    # Show dataframe preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Target column selection - show immediately after upload
    target_column = st.sidebar.selectbox("2️⃣ Select target column", options=df.columns)

# --- Run Pipeline button ---
if st.sidebar.button("🚀 Run Full Pipeline"):
    if df is not None and target_column is not None:
        st.info("🚀 Running pipeline... This may take a moment...")

        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()

        # OPTIONAL: You can modify ModelingAgent to accept dynamic target_column later
        orchestrator.run_pipeline()

        # Show agent outputs
        st.subheader("📢 LLM Preprocessing Suggestion")
        st.markdown(orchestrator.context.preprocessing_suggestion or "No suggestion available.")

        st.subheader("📢 LLM Modeling Suggestion")
        st.markdown(orchestrator.context.modeling_suggestion or "No suggestion available.")

        st.subheader("📢 LLM Final Report")
        st.markdown(orchestrator.context.final_report or "No report available.")
    else:
        st.warning("⚠️ Please upload CSV and select target column first.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using **Streamlit** and **llama-cpp-python**")
