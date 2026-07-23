from __future__ import annotations

import html

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
from glaurlex.ui.views.home import LOGO_WORDMARK_PATH, render_home
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

    # Logo de la aplicación en la parte superior de la barra lateral.
    if LOGO_WORDMARK_PATH.exists():
        st.sidebar.image(str(LOGO_WORDMARK_PATH), width="stretch")
        st.sidebar.divider()

    # Sidebar navigation (una sola vez -> evita duplicados)
    if is_multi_tenant():
        user = current_username() or "?"
        st.sidebar.success(f"Sesión: **{user}**")
    st.sidebar.header("Navegación")

    # Las etiquetas del radio se mantienen constantes entre reruns. Si cambiaran
    # (p. ej. añadiendo "(bloqueado)" cuando no hay dataset), al cargar un dataset
    # el conjunto de opciones se sustituiría y Streamlit reiniciaría el radio a su
    # índice 0 ("Inicio"), provocando un salto a la landing. El bloqueo real de las
    # páginas se aplica más abajo con `has_dataset_loaded()`.
    options = [
        "Inicio",
        "Carga de datos",
        "Variables",
        "Grupos",
        "Estadísticas por type",
        "Estadísticas por informante",
        "Grafos",
        "Visualización",
        "Inferencia",
    ]

    page_token_to_prefix = {
        "home": "Inicio",
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
        default="home",
        allowed_values=page_token_to_prefix.keys(),
    )
    requested_prefix = page_token_to_prefix.get(requested_token, "Inicio")
    default_page = next((opt for opt in options if opt.startswith(requested_prefix)), options[0])

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in options:
        st.session_state["nav_page"] = default_page

    # Navegación programática (p. ej. botón "Comenzar" de la landing). Se aplica
    # antes de instanciar el radio para poder fijar su clave de sesión.
    pending_nav = st.session_state.pop("_pending_nav", None)
    if pending_nav in options:
        st.session_state["nav_page"] = pending_nav

    page = st.sidebar.radio(
        "Ir a:",
        options,
        key="nav_page",
    )
    if not has_dataset_loaded():
        st.sidebar.caption(
            "🔒 Carga o selecciona un dataset en **Carga de datos** para desbloquear "
            "el resto de secciones."
        )
    current_token = next(
        (token for prefix, token in page_prefix_to_token.items() if page.startswith(prefix)),
        "load",
    )
    st.session_state["app::page_token"] = current_token
    sync_query_state(
        key="app::page_token",
        param="page",
        default="home",
        allowed_values=page_token_to_prefix.keys(),
    )

    if is_multi_tenant() and LOGOUT_URL:
        st.sidebar.markdown("<div style='flex:1 1 auto'></div>", unsafe_allow_html=True)
        # `st.link_button` fuerza target="_blank": abre el logout en una pestaña
        # nueva y deja la sesión original abierta. Renderizamos un ancla con
        # target="_self" para navegar en la misma pestaña y cerrar sesión aquí.
        logout_url = html.escape(LOGOUT_URL, quote=True)
        st.sidebar.markdown(
            f'<a href="{logout_url}" target="_self" class="glx-logout-btn" '
            'title="Cerrar sesión">Cerrar Sesión</a>'
            "<style>.glx-logout-btn{display:inline-flex;align-items:center;"
            "justify-content:center;padding:0.25rem 0.75rem;"
            "border:1px solid rgba(128,128,128,0.4);border-radius:0.5rem;"
            "color:inherit;text-decoration:none;font-weight:400;line-height:1.6;"
            "transition:color .15s,border-color .15s}"
            ".glx-logout-btn:hover{color:rgb(255,75,75);border-color:rgb(255,75,75)}"
            "</style>",
            unsafe_allow_html=True,
        )

    if page.startswith("Inicio"):
        render_home()
        return

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
