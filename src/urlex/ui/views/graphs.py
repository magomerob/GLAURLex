from __future__ import annotations

import streamlit as st

from urlex.ui.state import get_load_result


def render() -> None:
    st.header("3) Grafos (placeholder)")

    lr = get_load_result()
    if lr is None:
        st.warning("No hay datos cargados. Ve a **1) Carga de datos**.")
        return

    st.write(f"**Dataset activo:** `{lr.ref.name}`")
    st.info("Placeholder de grafos. Aquí irán los controles y el render del grafo.")
    st.button("Generar grafo (no hace nada aún)", disabled=True)
