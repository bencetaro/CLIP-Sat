import base64
import time
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests
import streamlit as st

# pd display settings
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

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
    r = requests.post(f"{api_base_url}/predict", params=params, json=payload, timeout=180)
    return r, (time.time() - start)

def show_inference_ui():
    st.title("Inference On CLIP-Sat Model")
    st.divider()

    api_base_url = _api_base_url()

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    runs, default_run = _get_runs(api_base_url)

    st.sidebar.header("Settings")
    run_names = [r["name"] for r in runs]
    selected_run = st.sidebar.selectbox(
        "Choose model (W&B run)",
        options=(run_names if run_names else ["(no runs found)"]),
        index=(run_names.index(default_run) if default_run in run_names else 0),
        disabled=not bool(run_names),
    )
    chosen_id = next((r["id"] for r in runs if r["name"] == selected_run), None)

    with st.sidebar.expander("Settings", expanded=False):
        top_k = st.slider("Top-K Class Filter", min_value=1, max_value=14, value=5)
        try_gpu = st.toggle("Try GPU (CUDA)", value=False, help="Default is CPU. Enable only if your server CUDA setup is correct.")
        # for this single image inference CPU is good enough anyway

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
            st.image(uploaded, width='stretch')
            image_b64 = base64.b64encode(uploaded.getvalue()).decode("utf-8")
    else:
        image_url = st.text_input("Image URL")
        if image_url:
            st.image(image_url, width='stretch')

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
        st.dataframe(top_df, width='stretch')

        labels = res["labels"]
        probs = res["probs"]
        st.session_state["last_prediction"] = {
            "predicted_label": res.get("predicted_label"),
            "latency": float(latency),
            "labels": labels,
            "probs": probs,
            "top_k": int(top_k),
        }
        dist = pd.DataFrame({"label": labels, "prob": probs}).sort_values("prob", ascending=False)
        st.subheader("Prediction distribution")
        st.bar_chart(dist.set_index("label")["prob"])

    last = st.session_state.get("last_prediction")
    if last and not run_btn:
        st.info(f"Last prediction: {last.get('predicted_label')}  (latency {last.get('latency', 0):.2f}s)")
        dist = pd.DataFrame({"label": last["labels"], "prob": last["probs"]}).sort_values("prob", ascending=False)
        st.subheader("Prediction distribution")
        st.bar_chart(dist.set_index("label")["prob"])

    if last:
        st.subheader("LLM review")
        llm_backend = st.selectbox(
            "Choose LLM backend",
            options=["Llama.cpp", "HuggingFace"],
            help="`Llama.cpp` is lighter and usually better for low RAM. `HuggingFace` is larger/slower but can be more capable.",
            width=200,
        )
        llm_backend_chosen = "hf_heavy" if llm_backend == "HuggingFace" else "llama_cpp"
        llm_use_gpu = st.toggle(
            "Use GPU",
            value=False,
            help="For Llama.cpp this enables GGUF GPU layers. For HuggingFace this requests CUDA.",
        )
        st.caption("Note: First call may be slow due to model load. If resources are limited, start with `Llama.cpp` on CPU.")

        # ## Perhaps not necessary (kept for debugging)
        # warmup_btn = st.button("Warm up selected LLM", key="llm_warmup_btn")
        # if warmup_btn:
        #     warmup_status = st.status("Warming up LLM…", expanded=False)
        #     try:
        #         wr = requests.post(
        #             f"{api_base_url}/llm/warmup",
        #             params={"backend": llm_backend_chosen, "use_gpu": llm_use_gpu},
        #             timeout=600,
        #         )
        #         if wr.status_code != 200:
        #             warmup_status.update(label="Warmup failed", state="error")
        #             st.error(wr.text)
        #         else:
        #             warmup_status.update(label="Warmup complete", state="complete")
        #             info = wr.json()
        #             st.info(
        #                 f"Loaded backend={info.get('backend')} "
        #                 f"gpu={info.get('use_gpu')} "
        #                 f"in {info.get('took_s', 0):.2f}s"
        #             )
        #     except Exception as e:
        #         warmup_status.update(label="Warmup failed", state="error")
        #         st.error(f"LLM warmup request failed: {e}")

        llm_btn = st.button("Generate LLM answer", type="secondary", key="llm_generate_btn")
        if llm_btn:
            llm_status = st.status("Generating LLM review…", expanded=False)
            try:
                payload = {
                    "labels": last["labels"],
                    "probs": last["probs"],
                    "top_k": last["top_k"],
                    "backend": llm_backend_chosen,
                    "use_gpu": llm_use_gpu,
                }
                rr = requests.post(f"{api_base_url}/llm/review", json=payload, timeout=600)
                if rr.status_code != 200:
                    llm_status.update(label="LLM failed", state="error")
                    st.error(rr.text)
                else:
                    llm_status.update(label="LLM complete", state="complete")
                    out = rr.json()
                    if out.get("warnings"):
                        st.warning("\n".join(out["warnings"]))
                    parsed = out.get("parsed")

                    if parsed:
                        with st.container(border=True):
                            for key, value in parsed.items():
                                field_name = key.replace("_", " ").title()
                                st.subheader(field_name)
                                if isinstance(value, list):
                                    if key == "guesses":
                                        st.write(", ".join(value))
                                    else:
                                        for item in value:
                                            st.markdown(f"- {item}")
                                elif isinstance(value, dict):
                                    for k, v in value.items():
                                        st.markdown(f"**{k}:** {v}")
                                else:
                                    st.write(value)
                    else:
                        st.text(out.get("raw", ""))
                    st.caption(
                        f"LLM backend={out.get('backend')} gpu={out.get('use_gpu')} "
                        f"took {out.get('took_s', 0):.2f}s"
                    )
            except Exception as e:
                llm_status.update(label="LLM failed", state="error")
                st.error(f"LLM request failed: {e}")

    st.divider()
    st.caption(
    """
    **Photo by [SpaceX](https://images.unsplash.com/photo-1460186136353-977e9d6085a1?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) on [Unsplash](https://unsplash.com).**
    """
    )
