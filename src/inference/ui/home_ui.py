import streamlit as st


def show_home_ui():
    st.title("CLIP-Sat 🛰️")    
    st.divider()

    st.caption("Satellite imagery classification with a finetuned CLIP model.")

    st.info(
        """
        Home page content comes here - description about the webapp / project.
        Also mention the dataset it is based on.
        KPI stuff. Anything else?
        """
    )

    st.markdown(
        """
        ### What’s in this demo
        - **Inference On CLIP-Sat Model**: Upload an image (or paste a URL) and predict the categories listed below.
        - **Training Results**: Browse W&B runs, their configs, summaries, and training statistics.
        - **Model Comparison (Not ready yet)**: Compare different experiments based on custom settings (metrics, parameters, runtime, etc.).
        """
    )

