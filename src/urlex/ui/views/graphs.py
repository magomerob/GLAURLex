from __future__ import annotations

import re

import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.dataset_service import DatasetService
from urlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    node_stats,
)
from urlex.core.graph_service import GraphService
from urlex.core.groups import ALL_GROUP, apply_group
from urlex.ui.state import ensure_groups_loaded_for_dataset, ensure_state


@st.cache_resource
def get_dataset_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_resource
def get_graph_service(processed_dir: str) -> GraphService:
    return GraphService(processed_dir)


@st.cache_data
def load_dataset(processed_dir: str, name: str):
    svc = get_dataset_service(processed_dir)
    return svc.load_processed(name)


@st.cache_data(show_spinner=False)
def compute_bigrams_cached(df_tema, cache_key: str):
    _ = cache_key
    return bigrams_for_tema(df_tema)


def _infer_informant_col(df_tema) -> str | None:
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


def _slug_graph_name(*parts: str) -> str:
    raw = "__".join(p for p in parts if p)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return name or "graph"


def render_graphs():
    s = ensure_state()
    ensure_groups_loaded_for_dataset(s.dataset_name)

    st.header("3) Grafos")
    st.write(f"Dataset activo: **{s.dataset_name}**")

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = load_dataset(processed_dir, s.dataset_name)

    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}
    if "active_group" not in st.session_state:
        st.session_state.active_group = "TODOS"
    if st.session_state.active_group not in st.session_state.groups:
        st.session_state.active_group = "TODOS"

    st.subheader("Grupo de informantes")
    group_names = list(st.session_state.groups.keys())
    active_group_name = st.selectbox(
        "Selecciona un grupo",
        group_names,
        index=group_names.index(st.session_state.active_group),
        key="graphs::group_select",
    )
    st.session_state.active_group = active_group_name
    group = st.session_state.groups[active_group_name]

    informantes_df = getattr(ds, "informantes", None)
    if informantes_df is None:
        st.warning("Este dataset no expone ds.informantes; no se podrá filtrar por grupos.")
        informantes_f = None
    else:
        informantes_f = apply_group(informantes_df, group)

    st.subheader("Tema")
    tema_names = sorted(ds.temas.keys())
    if not tema_names:
        st.warning("No hay temas disponibles en este dataset procesado.")
        return

    default_tema = st.session_state.get("graphs::tema", tema_names[0])
    if default_tema not in tema_names:
        default_tema = tema_names[0]
    tema = st.selectbox("Selecciona un tema", tema_names, index=tema_names.index(default_tema))
    st.session_state["graphs::tema"] = tema

    df_tema = ds.temas[tema]
    df_tema_f = df_tema
    informant_col = _infer_informant_col(df_tema)

    if informantes_f is not None and informant_col is not None:
        informant_id_col = (
            "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
        )
        if informant_id_col is None:
            allowed = set((informantes_f.index + 1).tolist())
        else:
            allowed = set(informantes_f[informant_id_col].tolist())
        df_tema_f = df_tema[df_tema[informant_col].isin(allowed)]
    elif informantes_f is not None and informant_col is None:
        st.info(
            "No he encontrado una columna de informante en el df del tema. "
            "No se aplica el filtro del grupo."
        )

    st.caption(
        f"Filas en tema **{tema}**: {len(df_tema):,} "
        + (
            f"→ tras grupo **{active_group_name}**: {len(df_tema_f):,}"
            if df_tema_f is not df_tema
            else ""
        )
    )

    st.subheader("Grafo")
    directed = st.toggle("Grafo dirigido", value=True, key="graphs::directed")
    graph = None
    graph_service = get_graph_service(processed_dir)
    graph_name = _slug_graph_name(tema, active_group_name, "dir" if directed else "und")
    graph_path = graph_service.get_graph_path(s.dataset_name, graph_name)

    if graph_path.exists():
        st.caption(f"Usando grafo guardado: `{graph_path.name}`")
        with st.spinner("Cargando grafo GML..."):
            graph = graph_service.load_graph(s.dataset_name, graph_name)
    else:
        cache_key = f"{s.dataset_name}::{tema}::{active_group_name}::{len(df_tema_f)}::{directed}"
        with st.spinner("Calculando bigramas..."):
            bigrams_df = compute_bigrams_cached(df_tema_f, cache_key=cache_key)
        graph = bigrams_to_dirgraph(bigrams_df) if directed else bigrams_to_undgraph(bigrams_df)
        try:
            graph_service.save_graph(s.dataset_name, graph_name, graph, overwrite=False)
            st.caption(f"Grafo guardado automáticamente: `{graph_path.name}`")
        except FileExistsError:
            st.caption(f"Grafo guardado automáticamente: `{graph_path.name}`")
        except Exception as exc:  # pragma: no cover - capa UI
            st.error(f"Error guardando el GML: {exc}")

    if graph is None:
        return

    st.subheader("Resumen del grafo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodos", f"{graph.number_of_nodes():,}")
    c2.metric("Aristas", f"{graph.number_of_edges():,}")
    c3.metric("Dirigido", "Sí" if graph.is_directed() else "No")

    st.subheader("Estadísticas por nodo")
    with st.spinner("Calculando estadísticas de nodos..."):
        stats_df = node_stats(graph)

    if len(stats_df) == 0:
        st.info("No hay nodos para mostrar.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        top_n = st.number_input("Top N", min_value=10, max_value=2000, value=50, step=10)
    with c2:
        query = st.text_input("Filtrar nodo (contiene)", value="")

    view = stats_df
    if query:
        view = view[view["node"].astype(str).str.contains(query, case=False, na=False)]
    view = view.head(int(top_n))

    column_help = {
        "node": "Etiqueta del nodo (token).",
        "degree": "Grado no ponderado (número de vecinos/aristas).",
        "degree_centrality": "Centralidad de grado (normalizada por NetworkX).",
        "strength": "Grado ponderado usando el peso `weight`.",
        "betweenness": "Centralidad de intermediación no normalizada (`normalized=False`).",
        "closeness": "Centralidad de cercanía con mejora Wasserman-Faust (`wf_improved=True`).",
        "pagerank": "PageRank ponderado (`weight='weight'`, `alpha=0.85`, `tol=1e-6`).",
        "eigenvector": "Centralidad de vector propio ponderada (`weight='weight'`, `max_iter=1000`).",
        "clustering": "Coeficiente de clustering ponderado (`weight='weight'`).",
        "in_degree": "Grado entrante no ponderado.",
        "out_degree": "Grado saliente no ponderado.",
        "in_strength": "Grado entrante ponderado (`weight='weight'`).",
        "out_strength": "Grado saliente ponderado (`weight='weight'`).",
    }

    column_config = {
        col: (
            st.column_config.TextColumn(col, help=column_help[col])
            if col == "node"
            else st.column_config.NumberColumn(col, help=column_help[col])
        )
        for col in view.columns
        if col in column_help
    }

    st.dataframe(view, width="stretch", hide_index=True, column_config=column_config)
    st.download_button(
        "Descargar CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{s.dataset_name}_{tema}_{active_group_name}_node_stats.csv",
        mime="text/csv",
    )

    gml_bytes = None
    if graph_path.exists():
        gml_bytes = graph_path.read_bytes()
    st.download_button(
        "Descargar GML",
        data=gml_bytes,
        file_name=graph_path.name,
        mime="text/plain",
        disabled=gml_bytes is None,
    )
