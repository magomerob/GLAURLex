from __future__ import annotations

import streamlit as st

from urlex.ui.state import ensure_state


def render_graphs():
    s = ensure_state()

    st.header("3) Grafos (placeholder)")
    st.info("Aquí irá la parte de grafos. De momento es un placeholder.")

    st.write(f"Dataset activo: **{s.dataset_name}**")

    st.subheader("Placeholder")
