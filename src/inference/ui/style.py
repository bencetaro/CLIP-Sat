from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from src.inference.clip_helpers import get_base64


def _guess_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def apply_global_style(
    *,
    main_bg_image: Optional[str] = None,
    sidebar_bg_image: Optional[str] = None,
    overlay_rgba: str = "rgba(10, 14, 30, 0.72)",
) -> None:
    """
    Apply global CSS overrides for the app.

    Notes:
    - Streamlit DOM attributes can change; selectors are written defensively.
    - If you pass background images, they are embedded as data URLs.
    """
    main_bg_css = ""
    if main_bg_image:
        main_b64 = get_base64(main_bg_image)
        main_mime = _guess_mime(main_bg_image)
        main_bg_css = f"""
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient({overlay_rgba}, {overlay_rgba}),
                url("data:{main_mime};base64,{main_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """

    sidebar_bg_css = ""
    if sidebar_bg_image:
        side_b64 = get_base64(sidebar_bg_image)
        side_mime = _guess_mime(sidebar_bg_image)
        sidebar_bg_css = f"""
        [data-testid="stSidebar"] > div:first-child {{
            background-image:
                linear-gradient({overlay_rgba}, {overlay_rgba}),
                url("data:{side_mime};base64,{side_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        """

        # opt in sidebar: width: 300px !important;

    st.markdown(
        f"""
        <style>
        /* Transparent header so background shows through */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* Main + sidebar backgrounds */
        {main_bg_css}
        {sidebar_bg_css}

        /* Rounded + glassy containers (experimental) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.10);
            background: rgba(18, 26, 51, 0.55);
            backdrop-filter: blur(8px);
        }}

        /* Inputs (BaseWeb) */
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {{
            border-radius: 0.9rem;
        }}

        /* Buttons */
        button {{
            border-radius: 999px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
