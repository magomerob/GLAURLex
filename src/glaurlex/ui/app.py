from __future__ import annotations

import streamlit as st

from glaurlex.config import LOGOUT_URL
from glaurlex.ui.state import (
    current_username,
    ensure_state,
    has_dataset_loaded,
    is_multi_tenant,
    sync_query_state,
)

# Páginas
from glaurlex.ui.views.charts import render_charts
from glaurlex.ui.views.graphs import render_graphs
from glaurlex.ui.views.grouping import render_grouping
from glaurlex.ui.views.inference import render_inference
from glaurlex.ui.views.informant_stats import render_informant_stats
from glaurlex.ui.views.load_data import render_load_data
from glaurlex.ui.views.variables import render_variables
from glaurlex.ui.views.visualize import render_visualize


def main():
    st.set_page_config(page_title="GLAURLex", layout="wide")

    ensure_state()

    st.set_page_config(
        page_title="GLAURLex",
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
    if is_multi_tenant():
        user = current_username() or "?"
        if LOGOUT_URL:
            info_col, logout_col = st.sidebar.columns([3, 1], vertical_alignment="center")
            info_col.success(f"Sesión: **{user}**")
            logout_col.link_button("Salir", LOGOUT_URL, help="Cerrar sesión")
        else:
            st.sidebar.success(f"Sesión: **{user}**")
    st.sidebar.header("Navegación")

    # Si no hay dataset, bloqueamos visualización y grafos en la UI
    if has_dataset_loaded():
        options = [
            "Carga de datos",
            "Variables",
            "Grupos",
            "Estadísticas por type",
            "Estadísticas por informante",
            "Grafos",
            "Visualización",
            "Inferencia",
        ]
    else:
        options = [
            "Carga de datos",
            "Variables (bloqueado)",
            "Grupos (bloqueado)",
            "Estadísticas por type (bloqueado)",
            "Estadísticas por informante (bloqueado)",
            "Grafos (bloqueado)",
            "Visualización (bloqueado)",
            "Inferencia (bloqueado)",
        ]

    page_token_to_prefix = {
        "load": "Carga de datos",
        "variables": "Variables",
        "groups": "Grupos",
        "stats": "Estadísticas por type",
        "informant_stats": "Estadísticas por informante",
        "graphs": "Grafos",
        "charts": "Visualización",
        "inference": "Inferencia",
    }
    page_prefix_to_token = {v: k for k, v in page_token_to_prefix.items()}

    requested_token = sync_query_state(
        key="app::page_token",
        param="page",
        default="load",
        allowed_values=page_token_to_prefix.keys(),
    )
    requested_prefix = page_token_to_prefix.get(requested_token, "Carga de datos")
    default_page = next((opt for opt in options if opt.startswith(requested_prefix)), options[0])

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in options:
        st.session_state["nav_page"] = default_page

    page = st.sidebar.radio(
        "Ir a:",
        options,
        key="nav_page",
    )
    current_token = next(
        (token for prefix, token in page_prefix_to_token.items() if page.startswith(prefix)),
        "load",
    )
    st.session_state["app::page_token"] = current_token
    sync_query_state(
        key="app::page_token",
        param="page",
        default="load",
        allowed_values=page_token_to_prefix.keys(),
    )

    if page.startswith("Carga de datos"):
        render_load_data()
        return

    # Gatekeeping duro (aunque el usuario fuerce la navegación)
    if not has_dataset_loaded():
        st.warning("Primero tienes que cargar o seleccionar un dataset en **1) Carga de datos**.")
        st.stop()

    if page.startswith("Grupos"):
        render_grouping()
    elif page.startswith("Variables"):
        render_variables()
    elif page.startswith("Estadísticas por informante"):
        render_informant_stats()
    elif page.startswith("Estadísticas por type"):
        render_visualize()
    elif page.startswith("Grafos"):
        render_graphs()
    elif page.startswith("Visualización"):
        render_charts()
    elif page.startswith("Inferencia"):
        render_inference()


if __name__ == "__main__":
    main()
