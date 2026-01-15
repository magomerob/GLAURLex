from __future__ import annotations

import streamlit as st
from urlex.core.models import LoadResult

STATE_KEY = "load_result"


def get_load_result() -> LoadResult | None:
    return st.session_state.get(STATE_KEY, None)


def set_load_result(lr: LoadResult | None) -> None:
    st.session_state[STATE_KEY] = lr


def has_data_loaded() -> bool:
    return get_load_result() is not None
