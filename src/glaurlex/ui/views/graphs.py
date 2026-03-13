from __future__ import annotations

import re

import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    community_leiden,
    connected_components_sorted,
    graph_stats,
    is_graph_connected,
    node_stats,
    small_world_indices,
)
from glaurlex.core.graph_service import GraphService
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.ui.state import (
    ensure_groups_loaded_for_dataset,
    ensure_state,
    sync_query_state,
)


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


def compute_communities(graph, cache_key: str) -> dict:
    state_key = f"comm_cache::{cache_key}"
    if state_key not in st.session_state:
        communities = community_leiden(graph, seed=42)
        st.session_state[state_key] = {
            node: i + 1 for i, community in enumerate(communities) for node in community
        }
    return st.session_state[state_key]


def _render_small_world_metric(column, label: str, value, error: str | None) -> None:
    if isinstance(value, (int, float)):
        column.metric(label, f"{value:.4f}")
        return

    column.metric(label, "N/A")
    if error:
        column.error(error)


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

    st.header("Grafos")
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

    with st.expander("Grupo y tema", expanded=True):
        st.subheader("Grupo de informantes")
        group_names = list(st.session_state.groups.keys())
        sync_query_state(
            key="graphs::group_select",
            param="g_group",
            default=st.session_state.active_group,
            allowed_values=group_names,
        )
        active_group_name = st.selectbox(
            "Selecciona un grupo",
            group_names,
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

        sync_query_state(
            key="graphs::tema",
            param="g_tema",
            default=st.session_state.get("graphs::tema", tema_names[0]),
            allowed_values=tema_names,
        )
        tema = st.selectbox(
            "Selecciona un tema",
            tema_names,
            key="graphs::tema",
        )

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
    sync_query_state(
        key="graphs::directed",
        param="g_directed",
        default=True,
        parse=lambda raw: raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        serialize=lambda value: "1" if value else "0",
    )
    directed = st.toggle("Grafo dirigido", key="graphs::directed")
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
        gml_bytes = None

    gml_bytes = None
    if graph_path.exists():
        gml_bytes = graph_path.read_bytes()

    actions_col1, actions_col2 = st.columns(2)
    with actions_col1:
        st.download_button(
            "Descargar GML",
            data=gml_bytes,
            file_name=graph_path.name,
            mime="text/plain",
            disabled=gml_bytes is None,
            use_container_width=True,
        )
    with actions_col2:
        recalc_clicked = st.button("Recalcular grafo", use_container_width=True)

    if recalc_clicked:
        with st.spinner("Recalculando grafo..."):
            bigrams_df = bigrams_for_tema(df_tema_f)
            graph = bigrams_to_dirgraph(bigrams_df) if directed else bigrams_to_undgraph(bigrams_df)
            try:
                graph_service.save_graph(s.dataset_name, graph_name, graph, overwrite=True)
                st.success(f"Grafo recalculado y guardado: `{graph_path.name}`")
                gml_bytes = graph_path.read_bytes()
            except Exception as exc:  # pragma: no cover - capa UI
                st.error(f"Error recalculando/guardando el GML: {exc}")
                return

    st.subheader("Resumen del grafo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodos", f"{graph.number_of_nodes():,}")
    c2.metric("Aristas", f"{graph.number_of_edges():,}")
    c3.metric("Dirigido", "Sí" if graph.is_directed() else "No")

    with st.spinner("Calculando estadísticas..."):
        stats_df = node_stats(graph)
        gstats = graph_stats(graph, stats_df, include_small_world=False)
        comm_cache_key = (
            f"{s.dataset_name}::{graph_name}::{graph.number_of_nodes()}::{graph.number_of_edges()}"
        )
        community_map = compute_communities(graph, cache_key=comm_cache_key)
        stats_df = stats_df.copy()
        stats_df["community_id"] = stats_df["node"].map(community_map).astype("Int64")

    if len(stats_df) == 0:
        st.info("No hay nodos para mostrar.")
        return

    st.subheader("Estadísticas generales")
    n_communities = stats_df["community_id"].nunique(dropna=True)
    gstats_view = {
        "Diámetro": [gstats["diameter"]],
        "Long. camino prom.": [gstats["avg_path_length"]],
        "Densidad": [gstats["density"]],
        "Componentes": [gstats["components"]],
        "Comunidades (Leiden)": [n_communities],
        "Grado prom.": [gstats["avg_degree"]],
        "Fuerza prom.": [gstats["avg_strength"]],
        "Clustering prom.": [gstats["avg_clustering"]],
    }
    gstats_help = {
        "Diámetro": "Diámetro de la mayor componente del grafo.",
        "Long. camino prom.": "Longitud media de caminos más cortos en la mayor componente.",
        "Densidad": "Densidad del grafo.",
        "Componentes": "Número de componentes conexas (o débilmente conexas si es dirigido).",
        "Comunidades (Leiden)": "Número de comunidades detectadas con el algoritmo de Leiden (ModularityVertexPartition, ponderado).",
        "Grado prom.": "Promedio de grado (sin pesos).",
        "Fuerza prom.": "Promedio de grado ponderado por `weight`.",
        "Clustering prom.": "Coeficiente de clustering promedio ponderado.",
    }
    gstats_column_config = {
        "Diámetro": st.column_config.NumberColumn("Diámetro", help=gstats_help["Diámetro"]),
        "Long. camino prom.": st.column_config.NumberColumn(
            "Long. camino prom.", help=gstats_help["Long. camino prom."], format="%.4f"
        ),
        "Densidad": st.column_config.NumberColumn(
            "Densidad", help=gstats_help["Densidad"], format="%.6f"
        ),
        "Componentes": st.column_config.NumberColumn(
            "Componentes", help=gstats_help["Componentes"]
        ),
        "Comunidades (Leiden)": st.column_config.NumberColumn(
            "Comunidades (Leiden)", help=gstats_help["Comunidades (Leiden)"]
        ),
        "Grado prom.": st.column_config.NumberColumn(
            "Grado prom.", help=gstats_help["Grado prom."], format="%.4f"
        ),
        "Fuerza prom.": st.column_config.NumberColumn(
            "Fuerza prom.", help=gstats_help["Fuerza prom."], format="%.4f"
        ),
        "Clustering prom.": st.column_config.NumberColumn(
            "Clustering prom.", help=gstats_help["Clustering prom."], format="%.4f"
        ),
    }
    st.dataframe(gstats_view, width="stretch", hide_index=True, column_config=gstats_column_config)

    st.subheader("Estadísticas por nodo")
    c1 = st.columns([1])[0]
    with c1:
        sync_query_state(
            key="graphs::top_n",
            param="g_top_n",
            default=50,
            parse=int,
            normalize=lambda v: max(10, min(2000, int(v))),
        )
        top_n = st.number_input(
            "Top N",
            min_value=10,
            max_value=2000,
            step=10,
            key="graphs::top_n",
        )

    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    strong_component_sizes = (
        stats_df.groupby("strong_component_id", dropna=True)["node"].size().to_dict()
    )
    strong_component_ids = stats_df["strong_component_id"].dropna().astype(int).unique().tolist()
    strong_component_ids.sort(
        key=lambda comp_id: (-strong_component_sizes.get(comp_id, 0), comp_id)
    )
    strong_component_options = ["Todos"] + strong_component_ids
    weak_component_options = ["Todos"] + sorted(
        stats_df["weak_component_id"].dropna().astype(int).unique().tolist()
    )
    community_sizes = stats_df.groupby("community_id", dropna=True)["node"].size().to_dict()
    community_ids = stats_df["community_id"].dropna().astype(int).unique().tolist()
    community_ids.sort(key=lambda cid: (-community_sizes.get(cid, 0), cid))
    community_options = ["Todos"] + community_ids

    sync_query_state(
        key="graphs::strong_component_filter",
        param="g_strong",
        default="Todos",
        parse=lambda raw: int(raw) if raw.strip().isdigit() else "Todos",
        allowed_values=strong_component_options,
    )
    sync_query_state(
        key="graphs::weak_component_filter",
        param="g_weak",
        default="Todos",
        parse=lambda raw: int(raw) if raw.strip().isdigit() else "Todos",
        allowed_values=weak_component_options,
    )
    sync_query_state(
        key="graphs::community_filter",
        param="g_comm",
        default="Todos",
        parse=lambda raw: int(raw) if raw.strip().isdigit() else "Todos",
        allowed_values=community_options,
    )
    sync_query_state(
        key="graphs::query",
        param="g_query",
        default="",
    )

    with f1:
        selected_strong_component = st.selectbox(
            "Comp. fuerte",
            strong_component_options,
            key="graphs::strong_component_filter",
        )
    with f2:
        selected_weak_component = st.selectbox(
            "Comp. débil",
            weak_component_options,
            key="graphs::weak_component_filter",
        )
    with f3:
        selected_community = st.selectbox(
            "Comunidad (Leiden)",
            community_options,
            key="graphs::community_filter",
        )
    with f4:
        query = st.text_input("Filtrar nodo (contiene)", key="graphs::query")

    view = stats_df
    if selected_strong_component != "Todos":
        view = view[view["strong_component_id"] == int(selected_strong_component)]
    if selected_weak_component != "Todos":
        view = view[view["weak_component_id"] == int(selected_weak_component)]
    if selected_community != "Todos":
        view = view[view["community_id"] == int(selected_community)]
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
        "strong_component_id": "ID de componente fuertemente conexa.",
        "weak_component_id": "ID de componente débilmente conexa.",
        "community_id": "ID de comunidad detectada con Leiden (ModularityVertexPartition, ponderado).",
        "in_degree": "Grado entrante no ponderado.",
        "out_degree": "Grado saliente no ponderado.",
        "in_strength": "Grado entrante ponderado (`weight='weight'`).",
        "out_strength": "Grado saliente ponderado (`weight='weight'`).",
        "SWI": "Small-worldness index. Valores mayores suelen indicar una estructura más small-world.",
        "ω": "Omega de small-world. Suele estar en el rango [-1, 1].",
    }

    hidden_columns = {"strong_component_id", "weak_component_id"}
    view_display = view.drop(columns=list(hidden_columns), errors="ignore")

    column_config = {
        col: (
            st.column_config.TextColumn(col, help=column_help[col])
            if col == "node"
            else st.column_config.NumberColumn(col, help=column_help[col])
        )
        for col in view_display.columns
        if col in column_help
    }

    st.dataframe(view_display, width="stretch", hide_index=True, column_config=column_config)
    st.download_button(
        "Descargar CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{s.dataset_name}_{tema}_{active_group_name}_node_stats.csv",
        mime="text/csv",
    )

    st.subheader("Índices Small-world")
    if graph.is_directed():
        st.info("Solo disponible para grafos no dirigidos.")
        return

    st.caption(
        "SWI está entre 0 y: valores mayores indican mayor nivel de small-world"
        "\n ω está entre -1 y 1: cerca de 0 sugiere small-world, cerca de -1 una "
        "red tipo lattice y cerca de 1 una red más aleatoria."
    )

    target_graph = None
    target_label = "grafo completo"
    component_suffix = "full"

    if graph.number_of_nodes() < 2:
        st.info("Se necesitan al menos 2 nodos para calcular índices small-world.")
    elif is_graph_connected(graph):
        target_graph = graph
    else:
        components = connected_components_sorted(graph)
        component_labels = [
            f"Componente {idx} ({len(nodes):,} nodos)"
            for idx, nodes in enumerate(components, start=1)
        ]
        selection = st.selectbox(
            "Componente conexa para small-world",
            ["Selecciona una componente"] + component_labels,
            key=f"graphs::sw_component::{graph_name}",
        )
        if selection != "Selecciona una componente":
            selected_idx = component_labels.index(selection) + 1
            selected_nodes = components[selected_idx - 1]
            target_graph = graph.subgraph(selected_nodes).copy()
            target_label = selection
            component_suffix = f"cc{selected_idx}"
            st.caption(f"Cálculo sobre: {selection}.")
        else:
            st.info(
                "El grafo no es conexo. Selecciona una componente conexa para habilitar el cálculo."
            )

    button_disabled = target_graph is None
    calculate_sw = st.button(
        "Calcular índices Small-world",
        use_container_width=True,
        disabled=button_disabled,
    )

    if target_graph is not None and not is_graph_connected(target_graph):
        st.error("La selección no es conexa; no se pueden calcular índices small-world.")
        return

    if target_graph is None:
        st.caption("El botón se habilita cuando haya un grafo conexo válido.")
        return

    sw_cache_key = (
        f"{s.dataset_name}::{graph_name}::{component_suffix}::"
        f"{target_graph.number_of_nodes()}::{target_graph.number_of_edges()}"
    )
    sw_state_key = f"graphs::small_world::{sw_cache_key}"

    if calculate_sw:
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress_cb(i: int, n: int) -> None:
            progress_bar.progress(0.0 if n <= 0 else min(i / n, 1.0))

        def status_cb(msg: str) -> None:
            status.write(msg)

        st.session_state[sw_state_key] = small_world_indices(
            target_graph,
            n=10,
            progress_cb=progress_cb,
            status_cb=status_cb,
        )
        progress_bar.progress(1.0)
        status.write("Cálculo completado.")

    small_world = st.session_state.get(sw_state_key, {})

    if not small_world:
        st.warning(
            "El cálculo de índices small-world puede tardar bastante tiempo en grafos grandes.",
            icon="⚠️",
        )

    if not small_world:
        st.caption("Pulsa el botón para calcular y mostrar SWI y ω.")
        return

    st.caption(f"Resultado mostrado para: {target_label}.")
    sw1, sw2 = st.columns(2)
    swi = small_world.get("SWI")
    swi_error = small_world.get("SWI_error")
    omega = small_world.get("ω")
    omega_error = small_world.get("ω_error")
    _render_small_world_metric(sw1, "SWI", swi, swi_error)
    _render_small_world_metric(sw2, "ω", omega, omega_error)
