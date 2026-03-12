from __future__ import annotations

import pandas as pd
import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.groups import ALL_GROUP, Group
from glaurlex.core.groups_store import save_groups
from glaurlex.ui.state import ensure_groups_loaded_for_dataset, ensure_state, has_dataset_loaded


@st.cache_resource
def get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def load_dataset(processed_dir: str, name: str):
    svc = get_service(processed_dir)
    return svc.load_processed(name)


def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    s2 = pd.to_numeric(s, errors="coerce")
    return s2.notna().mean() >= 0.9


def _fmt_rule(col: str, rule) -> str:
    if isinstance(rule, list):
        # compacta: recorta si hay muchos
        shown = rule[:3]
        tail = f" +{len(rule) - 3}" if len(rule) > 3 else ""
        return f"{col} IN {shown}{tail}"
    if isinstance(rule, dict) and "op" in rule:
        if rule["op"] == "between":
            return f"{col} ∈ [{rule.get('min')}, {rule.get('max')}]"
        return f"{col} {rule.get('op')} {rule.get('value')}"
    return f"{col} == {rule}"


def _summarize_filters(filters: dict) -> str:
    if not filters:
        return "—"
    parts = [_fmt_rule(c, r) for c, r in filters.items()]
    # aún más compacto
    s = " ∧ ".join(parts)
    return s if len(s) <= 120 else s[:117] + "…"


def render_grouping():
    s = ensure_state()
    st.header("Grupos")

    # Dataset requerido
    if not has_dataset_loaded():
        st.info("Carga un dataset para gestionar grupos.")
        return

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    # Cargar grupos asociados al dataset activo
    ensure_groups_loaded_for_dataset(s.dataset_name)

    # Asegurar TODOS
    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}
    if "TODOS" not in st.session_state.groups:
        st.session_state.groups["TODOS"] = ALL_GROUP
    if (
        "active_group" not in st.session_state
        or st.session_state.active_group not in st.session_state.groups
    ):
        st.session_state.active_group = "TODOS"

    # Selector grupo activo
    names = sorted(st.session_state.groups.keys(), key=lambda x: (x != "TODOS", x))
    """
    st.session_state.active_group = st.selectbox(
        "Grupo activo",
        names,
        index=names.index(st.session_state.active_group),
        key="grouping::active_group",
    )

    st.divider()
    """

    # Vista compacta: tabla de grupos
    st.subheader("Grupos existentes")

    rows = []
    for name in names:
        g = st.session_state.groups[name]
        rows.append(
            {
                "Nombre": name,
                "Tipo": "🔒" if g.immutable else "",
                "#filtros": len(getattr(g, "filters", {}) or {}),
                "Resumen": _summarize_filters(getattr(g, "filters", {}) or {}),
            }
        )
    df_view = pd.DataFrame(rows)

    st.dataframe(
        df_view,
        hide_index=True,
        width="stretch",
        column_config={
            "Nombre": st.column_config.TextColumn(width="medium"),
            "Tipo": st.column_config.TextColumn(width="small"),
            "#filtros": st.column_config.NumberColumn(width="small"),
            "Resumen": st.column_config.TextColumn(width="large"),
        },
    )

    # Eliminar grupo
    deletable = [
        n for n in names if not st.session_state.groups[n].immutable and n.upper() != "TODOS"
    ]

    with st.expander("Eliminar grupo", expanded=False):
        if not deletable:
            st.caption("No hay grupos eliminables.")
        else:
            to_delete = st.selectbox("Grupo", deletable, key="grouping::delete_select")

            if st.button("Eliminar", type="secondary", key="grouping::delete_btn"):
                del st.session_state.groups[to_delete]
                if st.session_state.active_group == to_delete:
                    st.session_state.active_group = "TODOS"
                save_groups(processed_dir, s.dataset_name, st.session_state.groups)
                st.success(f"Grupo '{to_delete}' eliminado.")
                st.rerun()

    st.divider()

    # Crear grupo
    st.subheader("Crear nuevo grupo (múltiples filtros)")
    new_name = st.text_input("Nombre del grupo", key="grouping::new_name").strip()

    ds = load_dataset(processed_dir, s.dataset_name)
    informantes = getattr(ds, "informantes", None)
    if informantes is None:
        st.warning("Este dataset procesado no contiene tabla de informantes (ds.informantes).")
        return

    drop_cols = {"user_id", "CODIGO_INFORMANTE", "codigoinformante", "codigo_informante"}
    cols = sorted([c for c in informantes.columns if c not in drop_cols])
    if not cols:
        st.warning("No hay variables disponibles para crear grupos.")
        return

    if "grouping::draft_filters" not in st.session_state:
        st.session_state["grouping::draft_filters"] = {}
    draft_filters: dict = st.session_state["grouping::draft_filters"]

    st.markdown("**Añadir filtro (se combinan con AND)**")
    col = st.selectbox("Variable", cols, key="grouping::draft_col")
    series = informantes[col]
    is_num = _is_numeric_series(series)

    if is_num:
        op = st.selectbox(
            "Operador", [">=", ">", "<=", "<", "==", "!=", "between"], key="grouping::draft_op"
        )
        s_num = pd.to_numeric(series, errors="coerce").dropna()

        if s_num.empty:
            st.error("La columna parece numérica pero no tiene valores numéricos parseables.")
            rule = None
        elif op == "between":
            c1, c2 = st.columns(2)
            with c1:
                vmin = st.number_input(
                    "Mínimo", value=float(s_num.min()), key="grouping::draft_min"
                )
            with c2:
                vmax = st.number_input(
                    "Máximo", value=float(s_num.max()), key="grouping::draft_max"
                )
            rule = None if vmin > vmax else {"op": "between", "min": vmin, "max": vmax}
            if vmin > vmax:
                st.error("El mínimo no puede ser mayor que el máximo.")
        else:
            val = st.number_input("Valor", value=float(s_num.median()), key="grouping::draft_val")
            rule = {"op": op, "value": val}
    else:
        raw_vals = series.dropna().unique().tolist()
        options = sorted(raw_vals, key=lambda x: str(x))
        vals = st.multiselect(
            "Valores (IN)", options=options, default=[], key="grouping::draft_vals"
        )
        rule = vals if vals else None

    if st.button("Añadir filtro", disabled=(rule is None), key="grouping::add_filter"):
        draft_filters[col] = rule
        st.session_state["grouping::draft_filters"] = draft_filters
        st.rerun()

    st.markdown("**Filtros del grupo (borrador)**")
    if not draft_filters:
        st.caption("Aún no has añadido filtros.")
    else:
        for fcol, frule in list(draft_filters.items()):
            c1, c2 = st.columns([8, 1])
            with c1:
                st.write("• " + _fmt_rule(fcol, frule))
            with c2:
                if st.button("🗑", key=f"grouping::del::{fcol}"):
                    draft_filters.pop(fcol, None)
                    st.session_state["grouping::draft_filters"] = draft_filters
                    st.rerun()

    c1, c2 = st.columns([1, 1])
    with c1:
        create_disabled = (not new_name) or (not draft_filters)
        if st.button("Crear grupo", disabled=create_disabled, key="grouping::create_group"):
            if new_name.upper() == "TODOS":
                st.error("TODOS está reservado.")
                return
            if new_name in st.session_state.groups:
                st.error("Ya existe un grupo con ese nombre.")
                return

            st.session_state.groups[new_name] = Group(
                name=new_name,
                filters=dict(draft_filters),
                immutable=False,
            )
            save_groups(processed_dir, s.dataset_name, st.session_state.groups)
            st.session_state["grouping::draft_filters"] = {}
            st.success(f"Grupo '{new_name}' creado.")
            st.rerun()

    with c2:
        if st.button("Limpiar borrador", key="grouping::clear_draft"):
            st.session_state["grouping::draft_filters"] = {}
            st.rerun()
