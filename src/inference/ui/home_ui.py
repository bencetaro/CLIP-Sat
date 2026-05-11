import streamlit as st
import os
import requests
from dotenv import load_dotenv
load_dotenv()

def _api_base_url() -> str:
    return os.getenv("INFERENCE_API_BASE_URL", "http://localhost:8000")

def show_home_ui():
    st.title("CLIP-Sat 🛰️")

    st.html("""
    <div class="caption-1">
        Satellite imagery classification with a finetuned CLIP model.
    </div>

    <style>
    .caption-1 {
        font-size: 20px;
        color: #a6a6a6;
        margin-bottom: 5px;
    }
    </style>
    """)

    st.divider()

    st.markdown(
        """
        ### What’s in this demo

        - **Make inference on a CLIP-Sat model:**
            - Upload an image (or paste a URL) and predict the following categories:
            - Agriculture, Airport, Beach, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, Residential, River.
        - **Inspect the training results:**
            - Browse W&B runs, their configs, summaries, and training statistics.
        - **Compare training runs (Not ready yet):**
            - Compare different experiments based on custom settings (metrics, parameters, runtime, etc.).

        ### About the training

        - This project is based on this following Kaggle dataset:
            - :violet-badge[[Source dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset/data)]
        - The data preprocessing, statistical overview and other specific details can be explored in this notebook:
            - :violet-badge[[Training notebook](https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning)]


        [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/bencetaro/CLIP-Sat)
        """
    )

    st.divider()
    st.subheader("Rate the application")
    with st.form("app_feedback_form", clear_on_submit=True):
        selected = st.feedback("stars")
        rating = 0 if selected is None else selected + 1
        comment = st.text_area("Leave your comments below (optional):", placeholder="What worked well? What should change?")
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        try:
            api_base_url = _api_base_url()
            r = requests.post(
                f"{api_base_url}/ui/feedback",
                json={"app_rating": int(rating), "app_comment": (comment or None)},
                timeout=5,
            )
            if r.status_code == 200:
                st.success("Thank you very much for your feedback!")
            else:
                st.error(f"Failed to save feedback: {r.text}")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

    st.divider()
    st.caption(
    """
    **Photo by [SpaceX](https://images.unsplash.com/photo-1460186136353-977e9d6085a1?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) on [Unsplash](https://unsplash.com).**
    """
    )
