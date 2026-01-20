from __future__ import annotations

import streamlit as st

from urlex.ui.state import ensure_state, has_dataset_loaded
from urlex.ui.views.graphs import render_graphs

# Páginas
from urlex.ui.views.load_data import render_load_data
from urlex.ui.views.visualize import render_visualize


def main():
    st.set_page_config(page_title="URLex", layout="wide")

    ensure_state()

    st.title("URLex")

    # No mostrar el botón de deploy
    st.markdown(
        r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation (una sola vez -> evita duplicados)
    st.sidebar.header("Navegación")

    # Si no hay dataset, bloqueamos visualización y grafos en la UI
    if has_dataset_loaded():
        options = ["1) Carga de datos", "2) Estadísticas", "3) Grafos"]
    else:
        options = ["1) Carga de datos", "2) Estadísticas (bloqueado)", "3) Grafos (bloqueado)"]

    page = st.sidebar.radio("Ir a:", options, index=0, key="nav_page")

    if page.startswith("1)"):
        render_load_data()
        return

    # Gatekeeping duro (aunque el usuario fuerce la navegación)
    if not has_dataset_loaded():
        st.warning("Primero tienes que cargar o seleccionar un dataset en **1) Carga de datos**.")
        st.stop()

    if page.startswith("2)"):
        render_visualize()
    elif page.startswith("3)"):
        render_graphs()


if __name__ == "__main__":
    main()
