from __future__ import annotations

import pandas as pd
import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.dataset_service import DatasetService
from urlex.core.graph import bigrams_for_tema, bigrams_to_unordered
from urlex.core.groups import ALL_GROUP, apply_group
from urlex.core.stats import estadisticas_df
from urlex.ui.state import ensure_groups_loaded_for_dataset, ensure_state


@st.cache_resource
def get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def load_dataset(processed_dir: str, name: str):
    svc = get_service(processed_dir)
    return svc.load_processed(name)


@st.cache_data(show_spinner=False)
def compute_stats_cached(df_tema, cache_key: str):
    # cache_key fuerza invalidación si cambias de tema/dataset/grupo
    _ = cache_key
    return estadisticas_df(df_tema)


@st.cache_data(show_spinner=False)
def compute_bigrams_cached(df_tema, cache_key: str):
    _ = cache_key
    return bigrams_for_tema(df_tema)


def _infer_informant_col(df_tema) -> str | None:
    """
    Intenta inferir la columna que identifica al informante dentro del df del tema.
    Ajusta esta lista según tu esquema real.
    """
    candidates = [
        "CODIGO_INFORMANTE",
        "codigoinformante",
        "codigo_informante",
        "informante",
        "user",
        "usuario",
        "center",
        "centers",
        "user_id",
    ]
    for c in candidates:
        if c in df_tema.columns:
            return c
    return None


def render_visualize():
    s = ensure_state()
    if "groups" not in st.session_state:
        ensure_groups_loaded_for_dataset(s.dataset_name)
    else:
        ensure_groups_loaded_for_dataset(s.dataset_name)
    st.header("Estadísticas")

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
    active_group_name = st.selectbox(
        "Selecciona un grupo",
        group_names,
        index=group_names.index(st.session_state.active_group),
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

    # recuerda selección
    default_tema = st.session_state.get("visualize::tema", tema_names[0])
    if default_tema not in tema_names:
        default_tema = tema_names[0]

    tema = st.selectbox("Selecciona un tema", tema_names, index=tema_names.index(default_tema))
    st.session_state["visualize::tema"] = tema

    df_tema = ds.temas[tema]

    # Filtrar
    df_tema_f = df_tema
    informant_col = _infer_informant_col(df_tema)

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
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        top_n = st.number_input("Top N", min_value=10, max_value=2000, value=50, step=10)
    with c2:
        min_ap = st.slider("Aparición mínima", 0.0, 1.0, 0.0, 0.01)
    with c3:
        query = st.text_input("Filtrar token (contiene)", value="")

    # Calcular estadísticas
    # cache_key para que el caché distinga dataset + tema + grupo + tamaño filtrado
    cache_key = f"{s.dataset_name}::{tema}::{active_group_name}::{len(df_tema_f)}"
    with st.spinner("Calculando estadísticas del tema..."):
        stats = compute_stats_cached(df_tema_f, cache_key=cache_key)

    # Filtros
    stats_view = stats
    if query:
        stats_view = stats_view[
            stats_view["token"].astype(str).str.contains(query, case=False, na=False)
        ]
    stats_view = stats_view[stats_view["aparición"] >= min_ap]

    stats_top = stats_view.head(int(top_n))

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tokens únicos", f"{len(stats):,}")
    k2.metric("Mostrados (tras filtros)", f"{len(stats_view):,}")
    k3.metric("Freq. Top N", f"{stats_top['freq_rel'].sum():.3f}" if len(stats_top) else "0.000")
    k4.metric("Disponibilidad máx", f"{stats['disponibilidad'].max():.4f}" if len(stats) else "—")

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
    unordered = st.toggle("Ignorar orden (a,b == b,a)", value=False)
    with st.spinner("Calculando bigramas del tema..."):
        bigrams_ordered = compute_bigrams_cached(df_tema_f, cache_key=cache_key)

    if unordered:
        bigrams_view = bigrams_to_unordered(bigrams_ordered)
    else:
        bigrams_view = bigrams_ordered.copy()

    if len(bigrams_view) == 0:
        bigrams_view = pd.DataFrame(columns=["token_1", "token_2", "aparición", "freq_rel"])
    else:
        bigrams_view = bigrams_view.rename(columns={"count": "aparición"})
        ninf = df_tema_f[informant_col].nunique() if informant_col in df_tema_f.columns else 1
        bigrams_view["freq_rel"] = bigrams_view["aparición"] / ninf if ninf > 0 else 0.0
        bigrams_view = bigrams_view.sort_values(
            ["aparición", "token_1", "token_2"], ascending=[False, True, True]
        )

    bigrams_top = bigrams_view.head(int(top_n))
    st.dataframe(
        bigrams_top,
        width="stretch",
        hide_index=True,
        column_config={
            "token_1": st.column_config.TextColumn("token_1"),
            "token_2": st.column_config.TextColumn("token_2"),
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
        plt.plot(stats_top["token"], stats_top["disponibilidad"])
        plt.xticks(rotation=90)
        plt.xlabel("token")
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
