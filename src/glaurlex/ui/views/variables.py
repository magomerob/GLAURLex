"""! @package glaurlex.ui.views.variables
Vista de exploración y configuración de variables sociolinguísticas
(detección de tipo, marcado como ordinales y orden de niveles).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.variables_store import save_variables
from glaurlex.ui.state import (
    ensure_state,
    ensure_variables_loaded_for_dataset,
    has_dataset_loaded,
)

# Reutilizamos las heurísticas de tipado de la vista de inferencia para
# evitar duplicar código y dependencias circulares (UI -> UI ya existe en el
# proyecto).
from glaurlex.ui.views.inference import (
    _INFORMANT_DROP_COLS,
    _categorize_variables,
    _is_numeric_high_card,
)


@st.cache_resource
def _get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def _load_dataset(processed_dir: str, name: str):
    return _get_service(processed_dir).load_processed(name)


def _summarize_order(order: List[str]) -> str:
    if not order:
        return "—"
    if len(order) <= 4:
        return " < ".join(order)
    return " < ".join(order[:3]) + f" … (+{len(order) - 3})"


def render_variables() -> None:
    s = ensure_state()
    st.header("Variables")
    st.caption(
        "Explora las variables sociolinguísticas del dataset y márcalas como "
        "**ordinales** definiendo el orden de sus niveles. Las variables "
        "ordinales se utilizan en inferencia para tests de tendencia "
        "(Spearman / Kendall) y respetan el orden en gráficos y descriptivos."
    )

    if not has_dataset_loaded():
        st.info("Carga un dataset para gestionar variables.")
        return

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ensure_variables_loaded_for_dataset(s.dataset_name)
    cfg: dict = st.session_state.get("variables", {}) or {}

    ds = _load_dataset(processed_dir, s.dataset_name)
    informantes = getattr(ds, "informantes", None)
    if informantes is None or informantes.empty:
        st.warning("Este dataset no contiene tabla de informantes.")
        return

    candidates = [c for c in informantes.columns if c not in _INFORMANT_DROP_COLS]
    cat_cols, _num_cols = _categorize_variables(informantes, candidates)

    # ---------------------------------------------------------------- Resumen
    st.subheader("Resumen de variables")
    rows = []
    for c in sorted(candidates):
        is_num = _is_numeric_high_card(informantes[c])
        tipo = "numérica continua" if is_num else "categórica"
        n_uni = int(informantes[c].dropna().nunique())
        col_cfg = cfg.get(c, {})
        ordinal = bool(col_cfg.get("ordinal"))
        order = list(col_cfg.get("order") or [])
        rows.append(
            {
                "Variable": c,
                "Tipo": tipo,
                "# niveles": n_uni,
                "Ordinal": "✅" if ordinal else "—",
                "Orden": _summarize_order(order) if ordinal else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if not cat_cols:
        st.info("No hay variables categóricas detectadas para marcar como ordinales.")
        return

    st.divider()

    # ----------------------------------------------------------- Editor
    st.subheader("Configurar variable como ordinal")
    sel_col = st.selectbox(
        "Variable categórica",
        cat_cols,
        key="variables::sel_col",
    )

    # Niveles únicos (orden alfabético por defecto)
    raw_levels = sorted(
        informantes[sel_col].dropna().astype(str).unique().tolist(), key=str
    )
    current_cfg = cfg.get(sel_col, {})
    is_ordinal = st.toggle(
        "Marcar como ordinal",
        value=bool(current_cfg.get("ordinal")),
        key=f"variables::ordinal::{sel_col}",
    )

    # Mantenemos un estado por columna para que los reordenamientos no se
    # pierdan entre rerenders.
    state_key = f"variables::draft_order::{sel_col}"
    saved_order = list(current_cfg.get("order") or [])
    if state_key not in st.session_state:
        # Inicializar con el orden guardado + cualquier nivel nuevo al final.
        seen = set(saved_order)
        st.session_state[state_key] = saved_order + [
            lv for lv in raw_levels if lv not in seen
        ]
    else:
        # Resincronizar si han aparecido niveles nuevos en datos.
        existing = set(st.session_state[state_key])
        for lv in raw_levels:
            if lv not in existing:
                st.session_state[state_key].append(lv)
        # Y eliminar niveles que ya no existen en los datos.
        st.session_state[state_key] = [
            lv for lv in st.session_state[state_key] if lv in raw_levels
        ]

    if is_ordinal:
        st.markdown(
            "**Orden de niveles** — usa las flechas ↑ / ↓ para reordenar "
            "(1 = primero). Cuando esté listo, pulsa **Guardar**."
        )

        levels_in_state: list[str] = st.session_state[state_key]

        def _swap(i: int, j: int) -> None:
            if 0 <= i < len(levels_in_state) and 0 <= j < len(levels_in_state):
                levels_in_state[i], levels_in_state[j] = (
                    levels_in_state[j],
                    levels_in_state[i],
                )
                st.session_state[state_key] = levels_in_state

        for idx, lv in enumerate(levels_in_state):
            col_pos, col_lv, col_up, col_down = st.columns([1, 8, 1, 1])
            with col_pos:
                st.markdown(f"**{idx + 1}.**")
            with col_lv:
                st.write(lv)
            with col_up:
                if st.button(
                    "↑",
                    key=f"variables::up::{sel_col}::{idx}",
                    disabled=(idx == 0),
                    help="Subir un puesto",
                ):
                    _swap(idx, idx - 1)
                    st.rerun()
            with col_down:
                if st.button(
                    "↓",
                    key=f"variables::down::{sel_col}::{idx}",
                    disabled=(idx == len(levels_in_state) - 1),
                    help="Bajar un puesto",
                ):
                    _swap(idx, idx + 1)
                    st.rerun()

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Guardar", type="primary", key=f"variables::save::{sel_col}"):
                ordered = list(st.session_state[state_key])
                cfg[sel_col] = {"ordinal": True, "order": ordered}
                save_variables(processed_dir, s.dataset_name, cfg)
                st.session_state["variables"] = cfg
                st.success(
                    f"Variable '{sel_col}' marcada como ordinal con "
                    f"{len(ordered)} niveles."
                )
                st.rerun()

        with c2:
            if st.button("Restablecer", key=f"variables::reset::{sel_col}"):
                cfg.pop(sel_col, None)
                save_variables(processed_dir, s.dataset_name, cfg)
                st.session_state["variables"] = cfg
                st.session_state.pop(state_key, None)
                st.success(f"Configuración de '{sel_col}' eliminada.")
                st.rerun()

        with c3:
            st.caption(f"Niveles detectados en datos: **{len(raw_levels)}**.")
    else:
        # Si el usuario desactiva el toggle, ofrecemos eliminar el registro
        # persistido (si existía).
        if sel_col in cfg:
            if st.button(
                "Quitar configuración ordinal",
                key=f"variables::unmark::{sel_col}",
            ):
                cfg.pop(sel_col, None)
                save_variables(processed_dir, s.dataset_name, cfg)
                st.session_state["variables"] = cfg
                st.success(f"'{sel_col}' ya no es ordinal.")
                st.rerun()
        else:
            st.caption(
                f"Niveles detectados: {', '.join(raw_levels[:8])}"
                + (" …" if len(raw_levels) > 8 else "")
            )
