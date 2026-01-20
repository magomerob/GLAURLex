from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppState:
    dataset_name: Optional[str] = None


def ensure_state():
    import streamlit as st

    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()

    return st.session_state["app_state"]


def has_dataset_loaded() -> bool:
    import streamlit as st

    s: AppState = st.session_state.get("app_state")
    return bool(s and s.dataset_name)
