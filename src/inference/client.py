import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.inference.ui.home_ui import show_home_ui
from src.inference.ui.inference_ui import show_inference_ui
from src.inference.ui.training_ui import show_training_ui
from src.inference.ui.style import apply_global_style

st.set_page_config(page_title="CLIP-Sat UI", layout="wide")

API_BASE_URL = os.getenv("INFERENCE_API_BASE_URL", "http://localhost:8000")

apply_global_style(
    main_bg_image="src/inference/background/spacex.jpg",
    sidebar_bg_image="src/inference/background/starry.png",
)

st.sidebar.title("Navigation")

nav_options = ["Home", "Inference", "Training"]
if hasattr(st.sidebar, "segmented_control"):
    page = st.sidebar.segmented_control("Go to", nav_options, default="Home")
else:
    page = st.sidebar.radio("Go to", nav_options)

def track(page_name):
    try:
        requests.post(f"{API_BASE_URL}/ui/page_view", params={"page": page_name})
    except:
        pass

if page == "Home":
    track("home")
    show_home_ui()

elif page == "Inference":
    track("inference")
    show_inference_ui()

elif page == "Training":
    track("training")
    show_training_ui()
