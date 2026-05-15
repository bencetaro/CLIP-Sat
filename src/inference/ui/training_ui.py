import os
from dotenv import load_dotenv
load_dotenv()

import base64
import re
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.inference.utils.wandb_helpers import parse_clf_report, parse_clf_report_split

class_mapping = {
    0: 'Agriculture',
    1: 'Airport',
    2: 'Beach',
    3: 'Desert',
    4: 'Forest',
    5: 'Grassland',
    6: 'Highway',
    7: 'Lake',
    8: 'Mountain',
    9: 'Parking',
    10: 'Port',
    11: 'Railway',
    12: 'Residential',
    13: 'River'
}

def _api_base_url() -> str:
    API_BASE_URL = os.getenv("INFERENCE_API_BASE_URL", "http://localhost:8000")
    return API_BASE_URL

def _decode_b64(data: str) -> bytes:
    return base64.b64decode(data)

def _map_class_label(label: str) -> str:
    s = str(label).strip()
    if s.isdigit():
        return class_mapping.get(int(s), s)
    return s

def show_training_ui():
    st.title("Training Results")
    st.divider()

    api_base_url = _api_base_url()

    # Model selector
    c1, c2 = st.columns([3,2])
    with c1:
        try:
            models = requests.get(f"{api_base_url}/models", timeout=10).json()
            runs = models.get("runs", [])
            default_run = models.get("default")
        except Exception as e:
            st.error(f"Failed to load runs from API: {e}")
            return

        if not runs:
            st.warning("No runs returned by the API. Check `WANDB_ENTITY`/`WANDB_PROJECT` in the backend.")
            return

        run_names = [r["name"] for r in runs]

        st.markdown("#### Select run ")
        selected_run = st.selectbox(
            "",
            run_names,
            index=(run_names.index(default_run) if default_run in run_names else 0),
            width=500,
        )
        chosen_id = next((r["id"] for r in runs if r["name"] == selected_run), None)

        # col1, col2, col3 = st.columns([2, 3, 10])
        # with col1:
        #     if st.button("Sync model artifact"): # (warm cache)
        #         r = requests.post(f"{api_base_url}/wandb/sync_model", params={"run_id": chosen_id}, timeout=120)
        #         if r.status_code == 200:
        #             st.success("Synced model artifact")
        #             st.json(r.json())
        #         else:
        #             st.error(r.text, r.status_code)

        # with col2:
        #     st.caption("Download latest `.pt` model for the run.", text_alignment="left")

    with c2:
        with st.container(border=False):
            st.markdown("### Side note")
            st.markdown(
                """
                The most significant differences can be measured in the following ways:

                - Models that only performed feature extraction were less able to recognize certain classes, or failed to recognize them altogether
                  (e.g., forest, agriculture, airport). This can be best observed in the **"Class distribution by metric"** section for a chosen metric.

                - This behavior is also reflected in the test accuracy of the models across runs:

                    - **focal-top-adam**: 0.94
                    - **focal-clfhead-adam**: 0.80
                    - **ce-top-adam**: 0.94
                    - **ce-clfhead-adam**: 0.75

                """
            )

    st.divider()

    # Training summary
    st.header("Training Summary")
    meta = requests.get(f"{api_base_url}/wandb/metadata", params={"run_id": chosen_id}, timeout=30).json()
    col1, col2 = st.columns([2, 2])
    with col1:
        with st.container(border=True):
            crtime_ = meta.get('creation', 'N/A').replace('T', ' ').split(':')
            crtime = crtime_[0]+':'+crtime_[1]
            st.write(f"Training creation time: {crtime}")
            st.write(f"Training runtime: {meta.get('runtime', 'N/A')} sec.")
            st.write("Training config:")
            config = meta.get("config", {})
            if config:
                config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
                st.dataframe(config_df, use_container_width=True)
            else:
                st.info("No config found for this run.")

    with col2:
        with st.container(border=True):
            st.write("Logged artifacts")
            artifacts = requests.get(f"{api_base_url}/wandb/artifacts", params={"run_id": chosen_id}, timeout=30).json()
            art_df = pd.DataFrame(artifacts.get("artifacts", []))
            if art_df.empty:
                st.info("No artifacts found for this run.")
            else:
                st.dataframe(art_df, use_container_width=True)

    # Show loss plot
    with st.container(border=True):
        st.subheader("Training curves")
        history = requests.get(f"{api_base_url}/wandb/history", params={"run_id": chosen_id}, timeout=30).json()
        df = pd.DataFrame(history)
        if df.empty:
            st.warning("No history returned (missing keys or empty run history).")
            return

        y_cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not y_cols:
            st.info(f"History keys: {list(df.columns)}")
            return

        fig = px.line(df, x=("epoch" if "epoch" in df.columns else df.index), y=y_cols)
        st.plotly_chart(fig, use_container_width=True)

    st.divider() # --------------------
    st.header("Evaluation On Test Set")

    # Classification report
    resp = requests.get(f"{api_base_url}/wandb/clf_report", params={"run_id": chosen_id}, timeout=30).json()
    html = resp.get("html")
    if not html:
        st.warning("Classification report not found for this run.")
        return
    match = re.search(r"<pre>(.*?)</pre>", html, re.S)
    if not match:
        st.warning("Classification report HTML does not contain a <pre> block.")
        return
    text = match.group(1)
    clf_df, summary_df = parse_clf_report_split(text)
    if not clf_df.empty:
        clf_df["class_name"] = clf_df["label"].apply(_map_class_label)
    else:
        clf_df["class_name"] = []

    col1, col2 = st.columns([3,2])
    with col1:
        with st.container(border=True):
            st.write("Classification report")
            st.dataframe(clf_df.style.highlight_max(subset=["f1-score"]))
        with st.container(border=True):
            st.write("Classification summary")
            st.table(summary_df)

    # Confusion Matrix
    with col2:
        with st.container(border=True):
            st.write("Confusion matrix")
            try:
                cm = requests.get(f"{api_base_url}/wandb/confusion_matrix", params={"run_id": chosen_id}, timeout=30).json()
                if cm.get("image_base64"):
                    st.image(_decode_b64(cm["image_base64"]), width=600)
                else:
                    st.info("No confusion matrix found for this run (expected a logged image with 'confusion' in the filename).")
            except Exception as e:
                st.warning(f"Failed to load confusion matrix: {e}")

    # Metric Bar-Chart
    with st.container(border=True):
        st.subheader("Class distribution by metric")
        metric = st.selectbox("Choose metric", ["f1-score", "precision", "recall", "support"], width=200)

        plot_df = clf_df[["class_name", metric]].copy()
        plot_df = plot_df.dropna()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.sort_values(metric, ascending=False)
        fig = px.bar(plot_df, x="class_name", y=metric)
        fig.update_layout(xaxis_title="Class", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption(
    """
    **Photo by [SpaceX](https://images.unsplash.com/photo-1460186136353-977e9d6085a1?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) on [Unsplash](https://unsplash.com).**
    """
    )
