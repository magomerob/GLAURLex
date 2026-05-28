from __future__ import annotations

import pandas as pd
import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.graph import bigrams_to_unordered
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.ui.metrics_cache import (
    bigrams_cached,
    infer_informant_col,
    type_stats_cached,
)
from glaurlex.ui.state import (
    ensure_groups_loaded_for_dataset,
    ensure_state,
    sync_query_state,
)


@st.cache_resource
def get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def load_dataset(processed_dir: str, name: str):
    svc = get_service(processed_dir)
    return svc.load_processed(name)


def render_visualize():
    s = ensure_state()
    if "groups" not in st.session_state:
        ensure_groups_loaded_for_dataset(s.dataset_name)
    else:
        ensure_groups_loaded_for_dataset(s.dataset_name)
    st.header("Estadísticas por type")

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = load_dataset(processed_dir, s.dataset_name)

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
        key="visualize::group_select",
        param="v_group",
        default=st.session_state.active_group,
        allowed_values=group_names,
    )
    active_group_name = st.selectbox(
        "Selecciona un grupo",
        group_names,
        key="visualize::group_select",
    )
    st.session_state.active_group = active_group_name
    group = st.session_state.groups[active_group_name]

    # Cargar y filtrar informantes (para contar / filtrar tema)
    informantes_df = getattr(ds, "informantes", None)
    if informantes_df is None:
        st.warning("Este dataset no expone ds.informantes; no se podrá filtrar por grupos.")
        informantes_f = None
    else:
        informantes_f = apply_group(informantes_df, group)

    with st.expander("Información del dataset", expanded=False):
        st.write(
            {
                "dataset": s.dataset_name,
                "processed_dir": processed_dir,
                "n_informantes_total": len(ds.informantes) if informantes_df is not None else None,
                "n_informantes_grupo": len(informantes_f) if informantes_f is not None else None,
                "grupo_activo": active_group_name,
                "n_temas": len(ds.temas),
            }
        )

    st.subheader("Temas")
    tema_names = sorted(ds.temas.keys())
    if not tema_names:
        st.warning("No hay temas disponibles en este dataset procesado.")
        return

    sync_query_state(
        key="visualize::tema",
        param="v_tema",
        default=st.session_state.get("visualize::tema", tema_names[0]),
        allowed_values=tema_names,
    )

    tema = st.selectbox(
        "Selecciona un tema",
        tema_names,
        key="visualize::tema",
    )

    df_tema = ds.temas[tema]

    # Filtrar
    df_tema_f = df_tema
    informant_col = infer_informant_col(df_tema)

    if informantes_f is not None and informant_col is not None:
        informant_id_col = (
            "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
        )
        if informant_id_col is None:
            # fallback: usar el index+1 si no hay columna explícita
            allowed = set((informantes_f.index + 1).tolist())
        else:
            allowed = set(informantes_f[informant_id_col].tolist())

        df_tema_f = df_tema[df_tema[informant_col].isin(allowed)]
    elif informantes_f is not None and informant_col is None:
        st.info(
            "No he encontrado una columna de informante en el df del tema "
            "(por ejemplo 'CODIGO_INFORMANTE' o 'centers'). No se aplica el filtro del grupo."
        )

    st.caption(
        f"Filas en tema **{tema}**: {len(df_tema):,} "
        + (
            f"→ tras grupo **{active_group_name}**: {len(df_tema_f):,}"
            if df_tema_f is not df_tema
            else ""
        )
    )

    # Controles
    sync_query_state(
        key="visualize::top_n",
        param="v_top_n",
        default=50,
        parse=int,
        normalize=lambda v: max(10, min(2000, int(v))),
    )
    sync_query_state(
        key="visualize::min_ap",
        param="v_min_ap",
        default=0.0,
        parse=float,
        normalize=lambda v: max(0.0, min(1.0, float(v))),
    )
    sync_query_state(
        key="visualize::query",
        param="v_query",
        default="",
    )
    sync_query_state(
        key="visualize::unordered",
        param="v_unordered",
        default=False,
        parse=lambda raw: raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        serialize=lambda value: "1" if value else "0",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        top_n = st.number_input(
            "Top N", min_value=10, max_value=2000, step=10, key="visualize::top_n"
        )
    with c2:
        min_ap = st.slider("Aparición mínima", 0.0, 1.0, key="visualize::min_ap", step=0.01)
    with c3:
        query = st.text_input("Filtrar type (contiene)", key="visualize::query")

    # Calcular estadísticas
    # cache_key para que el caché distinga dataset + tema + grupo + tamaño filtrado
    cache_key = f"{s.dataset_name}::{tema}::{active_group_name}::{len(df_tema_f)}"
    with st.spinner("Calculando estadísticas del tema..."):
        stats = type_stats_cached(df_tema_f, cache_key=cache_key)

    # Filtros
    stats_view = stats
    if query:
        stats_view = stats_view[
            stats_view["type"].astype(str).str.contains(query, case=False, na=False)
        ]
    stats_view = stats_view[stats_view["aparición"] >= min_ap]

    stats_top = stats_view.head(int(top_n))

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Types únicos", f"{len(stats):,}")
    k2.metric("Total tokens", f"{int(stats['tokens'].sum()):,}" if len(stats) else "0")
    k3.metric("Mostrados (tras filtros)", f"{len(stats_view):,}")
    k4.metric("Freq. Top N", f"{stats_top['freq_rel'].sum():.3f}" if len(stats_top) else "0.000")
    k5.metric("Disponibilidad máx", f"{stats['disponibilidad'].max():.4f}" if len(stats) else "—")

    st.divider()

    # Tabla principal
    st.subheader("Tabla de estadísticas (ordenada por disponibilidad)")
    st.dataframe(
        stats_top,
        width="stretch",
        hide_index=True,
        column_config={
            "disponibilidad": st.column_config.NumberColumn(format="%.6f"),
            "avg_pos": st.column_config.NumberColumn("avg_pos", format="%.3f"),
            "aparición": st.column_config.NumberColumn("aparición", format="%.4f"),
            "freq_rel": st.column_config.NumberColumn(format="%.6f"),
            "freq_acum": st.column_config.NumberColumn(format="%.6f"),
        },
    )

    # Descarga CSV
    st.download_button(
        "Descargar CSV (tras filtros)",
        data=stats_view.to_csv(index=False).encode("utf-8"),
        file_name=f"{s.dataset_name}_{tema}_{active_group_name}_estadisticas.csv",
        mime="text/csv",
    )

    st.divider()

    # Tabla de bigramas
    st.subheader("Tabla de bigramas (ordenada por aparición)")
    unordered = st.toggle("Ignorar orden (a,b == b,a)", key="visualize::unordered")
    with st.spinner("Calculando bigramas del tema..."):
        bigrams_ordered = bigrams_cached(df_tema_f, cache_key=cache_key)

    if unordered:
        bigrams_view = bigrams_to_unordered(bigrams_ordered)
    else:
        bigrams_view = bigrams_ordered.copy()

    if len(bigrams_view) == 0:
        bigrams_view = pd.DataFrame(columns=["type_1", "type_2", "aparición", "freq_rel"])
    else:
        bigrams_view = bigrams_view.rename(columns={"count": "aparición"})
        ninf = df_tema_f[informant_col].nunique() if informant_col in df_tema_f.columns else 1
        bigrams_view["freq_rel"] = bigrams_view["aparición"] / ninf if ninf > 0 else 0.0
        bigrams_view = bigrams_view.sort_values(
            ["aparición", "type_1", "type_2"], ascending=[False, True, True]
        )

    bigrams_top = bigrams_view.head(int(top_n))
    st.dataframe(
        bigrams_top,
        width="stretch",
        hide_index=True,
        column_config={
            "type_1": st.column_config.TextColumn("type_1"),
            "type_2": st.column_config.TextColumn("type_2"),
            "aparición": st.column_config.NumberColumn("aparición", format="%.0f"),
            "freq_rel": st.column_config.NumberColumn("freq_rel", format="%.6f"),
        },
    )

    """
    # Gráficos
    st.subheader("Gráficos")

    if len(stats_top) > 0:
        # 1) Disponibilidad (Top N)
        fig1 = plt.figure()
        plt.plot(stats_top["type"], stats_top["disponibilidad"])
        plt.xticks(rotation=90)
        plt.xlabel("type")
        plt.ylabel("disponibilidad")
        plt.tight_layout()
        st.pyplot(fig1, clear_figure=True)

    # 2) Frecuencia acumulada (sobre ranking por disponibilidad)
    fig2 = plt.figure()
    plt.plot(stats["freq_acum"].to_numpy() if len(stats) else [])
    plt.xlabel("rank (por disponibilidad)")
    plt.ylabel("freq_acum")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    """
