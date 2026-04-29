import streamlit as st

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
            - [Source dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset/data)
        - The data preprocessing, statistical overview and other specific details can be explored in this notebook:
            - [Training notebook](https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning)

        """
    )

    st.markdown(
        """
        :violet-badge[📝 Note]
        This is still a work in progress... There are a couple of more features to add...

        [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/bencetaro/CLIP-Sat)
        """
    )


    st.divider()
    st.caption(
    """
    **Photo by [SpaceX](https://images.unsplash.com/photo-1460186136353-977e9d6085a1?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) on [Unsplash](https://unsplash.com).**
    """
    )
