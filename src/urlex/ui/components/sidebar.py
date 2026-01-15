from __future__ import annotations

import streamlit as st


def render_sidebar(available_pages: list[str]) -> str:
    """
    Si ya tenéis una sidebar propia, sustituye el cuerpo por vuestra lógica,
    pero mantén la idea: recibir `available_pages` y devolver la página elegida.
    """
    st.sidebar.title("Navegación")

    # Guardamos la selección para que sea estable entre reruns
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = available_pages[0]

    # Si la selección anterior ya no es válida (p.ej. se descargaron datos),
    # la forzamos a la primera disponible.
    if st.session_state.selected_page not in available_pages:
        st.session_state.selected_page = available_pages[0]

    page = st.sidebar.radio(
        "Secciones",
        options=available_pages,
        index=available_pages.index(st.session_state.selected_page),
    )

    st.session_state.selected_page = page
    return page
