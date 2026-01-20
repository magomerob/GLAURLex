from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.dataset_service import DatasetService
from urlex.core.stats import estadisticas_df  # <- usa la versión vectorizada
from urlex.ui.state import ensure_state


@st.cache_resource
def get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def load_dataset(processed_dir: str, name: str):
    svc = get_service(processed_dir)
    return svc.load_processed(name)


@st.cache_data(show_spinner=False)
def compute_stats_cached(df_tema, cache_key: str):
    # cache_key fuerza invalidación si cambias de tema/dataset
    _ = cache_key
    return estadisticas_df(df_tema)


def render_visualize():
    s = ensure_state()
    st.header("Estadísticas")

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = load_dataset(processed_dir, s.dataset_name)

    # Opcional: pequeño contexto del dataset
    with st.expander("Información del dataset", expanded=False):
        st.write(
            {
                "dataset": s.dataset_name,
                "processed_dir": processed_dir,
                "n_informantes": len(ds.informantes),
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

    st.caption(f"Filas en tema **{tema}**: {len(df_tema):,}")

    # Controles
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        top_n = st.number_input("Top N", min_value=10, max_value=2000, value=50, step=10)
    with c2:
        min_ap = st.slider("Aparición mínima", 0.0, 1.0, 0.0, 0.01)
    with c3:
        query = st.text_input("Filtrar token (contiene)", value="")

    # Calcular estadísticas
    # cache_key para que el caché distinga dataset + tema
    cache_key = f"{s.dataset_name}::{tema}::{len(df_tema)}"
    with st.spinner("Calculando estadísticas del tema..."):
        stats = compute_stats_cached(df_tema, cache_key=cache_key)

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
    k3.metric("Freq. Top N", f"{stats_top['freq_rel'].sum():.3f}")
    k4.metric("Disponibilidad máx", f"{stats['disponibilidad'].max():.4f}")

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
        file_name=f"{s.dataset_name}_{tema}_estadisticas.csv",
        mime="text/csv",
    )

    st.divider()

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
    plt.plot(stats["freq_acum"].to_numpy())
    plt.xlabel("rank (por disponibilidad)")
    plt.ylabel("freq_acum")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
