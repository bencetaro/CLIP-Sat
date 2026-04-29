import base64
import time
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests
import streamlit as st


def _api_base_url() -> str:
    API_BASE_URL = os.getenv("INFERENCE_API_BASE_URL", "http://localhost:8000")
    return API_BASE_URL

def _get_runs(api_base_url: str):
    try:
        data = requests.get(f"{api_base_url}/models", timeout=5).json()
        runs = data.get("runs", [])
        default = data.get("default")
        return runs, default
    except Exception:
        return [], None

def _predict(api_base_url: str, run_id: Optional[str], image_b64: Optional[str], image_url: Optional[str], top_k: int, try_gpu: bool):
    payload = {"run_id": run_id, "image_base64": image_b64, "image_url": image_url, "top_k": top_k}
    start = time.time()
    params = {"device": "cuda"} if try_gpu else None
    r = requests.post(f"{api_base_url}/predict", params=params, json=payload, timeout=120)
    return r, (time.time() - start)

def show_inference_ui():
    st.title("Inference On CLIP-Sat Model")
    st.divider()

    api_base_url = _api_base_url()

    runs, default_run = _get_runs(api_base_url)

    st.sidebar.header("Settings")
    run_names = [r["name"] for r in runs]
    selected_run = st.sidebar.selectbox(
        "Model (W&B run name)",
        options=(run_names if run_names else ["(no runs found)"]),
        index=(run_names.index(default_run) if default_run in run_names else 0),
        disabled=not bool(run_names),
    )
    chosen_id = next((r["id"] for r in runs if r["name"] == selected_run), None)

    with st.sidebar.expander("Settings", expanded=False):
        top_k = st.slider("Top-K Class Filter", min_value=1, max_value=14, value=3)
        try_gpu = st.toggle("Try GPU (CUDA)", value=False, help="Default is CPU. Enable only if your server CUDA setup is correct.")

    try:
        r = requests.get(f"{api_base_url}/health", timeout=3)
        st.sidebar.success("API healthy" if r.status_code == 200 else "API error")
    except Exception:
        st.sidebar.error("API unreachable")

    input_mode = st.radio("Image input", ["Upload file", "From URL"], horizontal=True)

    image_b64 = None
    image_url = None

    if input_mode == "Upload file":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            st.image(uploaded, use_container_width=True)
            image_b64 = base64.b64encode(uploaded.getvalue()).decode("utf-8")
    else:
        image_url = st.text_input("Image URL")
        if image_url:
            st.image(image_url, use_container_width=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Predict", type="primary", disabled=not (image_b64 or image_url))
    with col2:
        st.caption("Tip: warm the backend by selecting a run and hitting Predict once (it will download the W&B model artifact).")

    if run_btn:
        run_id = chosen_id if run_names else None
        status = st.status("Running inference…", expanded=False)
        try:
            with st.spinner("Sending request to API…"):
                r, latency = _predict(api_base_url, run_id, image_b64, image_url, top_k, try_gpu=try_gpu)
            if r.status_code != 200:
                status.update(label="Inference failed", state="error")
                st.error(r.text)
                return
            status.update(label="Inference complete", state="complete")
        finally:
            pass

        res = r.json()
        st.success(f"Predicted: {res['predicted_label']}  (latency {latency:.2f}s)")

        top_df = pd.DataFrame(res["top_k"])
        top_df["score"] = top_df["score"].astype(float)
        st.subheader("Top predictions")
        st.dataframe(top_df, use_container_width=True)

        labels = res["labels"]
        probs = res["probs"]
        dist = pd.DataFrame({"label": labels, "prob": probs}).sort_values("prob", ascending=False)
        st.subheader("Prediction distribution")
        st.bar_chart(dist.set_index("label")["prob"])

    st.divider()
    st.caption(
    """
    **Photo by [SpaceX](https://images.unsplash.com/photo-1460186136353-977e9d6085a1?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) on [Unsplash](https://unsplash.com).**
    """
    )
