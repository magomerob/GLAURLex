from __future__ import annotations

import streamlit as st

from urlex.ui.state import ensure_state, has_dataset_loaded

# Páginas
from urlex.ui.views.graphs import render_graphs
from urlex.ui.views.grouping import render_grouping
from urlex.ui.views.load_data import render_load_data
from urlex.ui.views.visualize import render_visualize


def main():
    st.set_page_config(page_title="URLex", layout="wide")

    ensure_state()

    st.set_page_config(
        page_title="URLex",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
        options = ["Carga de datos", "Grupos", "Estadísticas", "Grafos"]
    else:
        options = [
            "Carga de datos",
            "Grupos (bloqueado)",
            "Estadísticas (bloqueado)",
            "Grafos (bloqueado)",
        ]

    page = st.sidebar.radio("Ir a:", options, index=0, key="nav_page")

    if page.startswith("Carga de datos"):
        render_load_data()
        return

    # Gatekeeping duro (aunque el usuario fuerce la navegación)
    if not has_dataset_loaded():
        st.warning("Primero tienes que cargar o seleccionar un dataset en **1) Carga de datos**.")
        st.stop()

    if page.startswith("Grupos"):
        render_grouping()
    elif page.startswith("Estadísticas"):
        render_visualize()
    elif page.startswith("Grafos"):
        render_graphs()


if __name__ == "__main__":
    main()
