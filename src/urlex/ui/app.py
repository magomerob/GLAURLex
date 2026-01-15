from __future__ import annotations

import streamlit as st

from urlex.ui.components.sidebar import render_sidebar
from urlex.ui.state import has_data_loaded
from urlex.ui.views import graphs, load_data, visualize

APP_TITLE = "URLex - UI"

PAGE_LOAD = "1) Carga de datos"
PAGE_VIZ = "2) Visualización"
PAGE_GRAPHS = "3) Grafos"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if has_data_loaded():
        available_pages = [PAGE_LOAD, PAGE_VIZ, PAGE_GRAPHS]
    else:
        available_pages = [PAGE_LOAD]

    page = render_sidebar(available_pages)

    # Guard extra
    if not has_data_loaded() and page in (PAGE_VIZ, PAGE_GRAPHS):
        st.warning("Primero carga datos para acceder a esta sección.")
        page = PAGE_LOAD

    if page == PAGE_LOAD:
        load_data.render()
    elif page == PAGE_VIZ:
        visualize.render()
    elif page == PAGE_GRAPHS:
        graphs.render()
    else:
        st.error(f"Página desconocida: {page}")


if __name__ == "__main__":
    main()
