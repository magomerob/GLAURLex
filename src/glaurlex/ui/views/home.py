"""! @package glaurlex.ui.views.home
Página de inicio (landing) que se muestra antes de la carga de datos.

Presenta la identidad de la aplicación (nombre y descripción), la financiación
del proyecto y la autoría. Incluye un botón para saltar directamente a la
sección de carga de datos.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st

# Assets empaquetados dentro de `glaurlex/ui/assets`
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_WORDMARK_PATH = _ASSETS_DIR / "logo_glaur_wordmark.png"
MINISTERIO_PATH = _ASSETS_DIR / "ministerio.png"

AUTHOR_EMAIL = "magomerob@unirioja.es"

def _app_version() -> str:
    try:
        return version("glaurlex")
    except PackageNotFoundError:
        return "0.1.0"


def render_home() -> None:
    """! Renderiza la landing con nombre, descripción, financiación y autoría."""
    st.markdown(
        """
        <style>
        .glx-hero { text-align: center; margin-top: 2rem; }
        .glx-title {
            font-size: 3.2rem; font-weight: 800; letter-spacing: .04em;
            margin: .25rem 0 0 0; line-height: 1.05;
        }
        .glx-tagline {
            font-size: 1.15rem; opacity: .75; margin: .35rem 0 0 0;
        }
        .glx-lead {
            font-size: 1.02rem; line-height: 1.6; opacity: .9;
            margin: 1.4rem 0 0 0; text-align: center;
        }
        .glx-funding {
            text-align: center; opacity: .8; font-size: .92rem;
            margin: .25rem 0 1rem 0; line-height: 1.5;
        }
        .glx-authorship {
            text-align: center; opacity: .7; font-size: .92rem; margin-top: .35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Todo el contenido vive en una única columna central para que los textos,
    # el título, el botón y la imagen queden centrados y alineados en el mismo eje.
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<div class='glx-hero'>"
            "<div class='glx-title'>GLAURLex</div>"
            "<div class='glx-tagline'>Análisis de disponibilidad léxica</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p class='glx-lead'>"
            "Herramienta para el estudio sociolingüístico de la disponibilidad léxica: "
            "cálculo de estadísticos "
            "por type o informante, construcción y análisis de grafos de co-ocurrencia por tema, y "
            "estadística descriptiva e inferencial frente a los metadatos de los informantes."
            "</p>",
            unsafe_allow_html=True,
        )

        st.write("")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Comenzar  →", type="primary", width="stretch"):
                st.session_state["_pending_nav"] = "Carga de datos"
                st.rerun()

        st.divider()

        # Financiación: texto del proyecto + banner institucional.
        st.markdown(
            "<p class='glx-funding'>"
            "Esta herramienta ha sido desarrollada por Marcos Gómez Robres como parte del Proyecto "
            "<strong>PID2022-137337NB-C21</strong> "
            "MICIU/AEI/10.13039/501100011033/ y FEDER/EU."
            "</p>",
            unsafe_allow_html=True,
        )
        if MINISTERIO_PATH.exists():
            st.image(str(MINISTERIO_PATH), width="stretch")

        st.divider()
        st.markdown(
            "<div class='glx-authorship'>"
            f"<a href='mailto:{AUTHOR_EMAIL}'>{AUTHOR_EMAIL}</a><br>"
            f"GLAURLex v{_app_version()}"
            "</div>",
            unsafe_allow_html=True,
        )
