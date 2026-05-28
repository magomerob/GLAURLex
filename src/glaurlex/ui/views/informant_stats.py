from __future__ import annotations

import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.core.metrics_catalog import labels_by_scope
from glaurlex.ui.metrics_cache import (
    filter_by_group,
    infer_informant_col,
    informant_metrics_cached,
)
from glaurlex.ui.state import (
    ensure_groups_loaded_for_dataset,
    ensure_state,
    sync_query_state,
)


INFORMANT_METRIC_COLS: dict[str, str] = labels_by_scope("informant")


@st.cache_resource
def _get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def _load_dataset(processed_dir: str, name: str):
    return _get_service(processed_dir).load_processed(name)


def render_informant_stats():
    s = ensure_state()
    ensure_groups_loaded_for_dataset(s.dataset_name)
    st.header("Estadísticas por informante")
    st.caption(
        "Métricas léxicas calculadas por informante (las mismas que se utilizan "
        "en la pestaña de Inferencia)."
    )

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = _load_dataset(processed_dir, s.dataset_name)
    informantes_df = getattr(ds, "informantes", None)

    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}
    if "active_group" not in st.session_state:
        st.session_state.active_group = "TODOS"
    if st.session_state.active_group not in st.session_state.groups:
        st.session_state.active_group = "TODOS"

    # Selector de grupo
    st.subheader("Grupo de informantes")
    group_names = list(st.session_state.groups.keys())
    sync_query_state(
        key="informant_stats::group_select",
        param="is_group",
        default=st.session_state.active_group,
        allowed_values=group_names,
    )
    active_group_name = st.selectbox(
        "Selecciona un grupo",
        group_names,
        key="informant_stats::group_select",
    )
    st.session_state.active_group = active_group_name
    group = st.session_state.groups[active_group_name]

    informantes_f = apply_group(informantes_df, group) if informantes_df is not None else None

    if informantes_df is None:
        st.warning("Este dataset no contiene tabla de informantes.")
        return

    # Selector de tema
    st.subheader("Tema")
    tema_names = sorted(ds.temas.keys())
    if not tema_names:
        st.warning("No hay temas disponibles en este dataset procesado.")
        return

    sync_query_state(
        key="informant_stats::tema",
        param="is_tema",
        default=st.session_state.get("informant_stats::tema", tema_names[0]),
        allowed_values=tema_names,
    )
    tema = st.selectbox(
        "Selecciona un tema",
        tema_names,
        key="informant_stats::tema",
    )

    df_tema = ds.temas[tema]
    informant_col = infer_informant_col(df_tema)
    df_tema_f = filter_by_group(df_tema, informantes_f, informant_col)

    st.caption(
        f"Filas en tema **{tema}**: {len(df_tema):,} "
        + (
            f"→ tras grupo **{active_group_name}**: {len(df_tema_f):,}"
            if df_tema_f is not df_tema
            else ""
        )
    )

    if df_tema_f.empty:
        st.warning("No hay datos para este tema/grupo.")
        return

    cache_key = f"{s.dataset_name}::{tema}::{active_group_name}::{len(df_tema_f)}"
    with st.spinner("Calculando métricas por informante..."):
        metrics_df = informant_metrics_cached(
            df_tema_f,
            informantes_f if informantes_f is not None else informantes_df,
            cache_key=cache_key,
        )

    if metrics_df.empty:
        st.warning("No se pudieron calcular métricas para este tema/grupo.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Informantes", f"{len(metrics_df):,}")
    k2.metric(
        "Tokens medios",
        f"{metrics_df['n_tokens'].mean():.2f}" if "n_tokens" in metrics_df else "—",
    )
    k3.metric(
        "Types medios",
        f"{metrics_df['n_types'].mean():.2f}" if "n_types" in metrics_df else "—",
    )
    k4.metric(
        "TTR medio",
        f"{metrics_df['ttr'].mean():.4f}" if "ttr" in metrics_df else "—",
    )

    st.divider()

    # Reordenar columnas: identificador + métricas + variables sociolinguísticas
    metric_cols = [c for c in INFORMANT_METRIC_COLS if c in metrics_df.columns]
    id_cols = [c for c in ("user_id", "CODIGO_INFORMANTE") if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in metric_cols and c not in id_cols]
    ordered_cols = id_cols + metric_cols + other_cols
    table = metrics_df[ordered_cols]

    st.subheader("Tabla de métricas por informante")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "type_coverage": st.column_config.NumberColumn("type_coverage", format="%.4f"),
            "ttr": st.column_config.NumberColumn("ttr", format="%.4f"),
            "mean_pos": st.column_config.NumberColumn("mean_pos", format="%.3f"),
            "entropy": st.column_config.NumberColumn("entropy", format="%.4f"),
            "total_disp": st.column_config.NumberColumn("total_disp", format="%.6f"),
            "mean_disp": st.column_config.NumberColumn("mean_disp", format="%.6f"),
        },
    )

    st.download_button(
        "Descargar CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{s.dataset_name}_{tema}_{active_group_name}_metricas_informante.csv",
        mime="text/csv",
    )
