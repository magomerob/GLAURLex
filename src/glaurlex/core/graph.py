"""! @package glaurlex.core.graph
Utilidades de grafos para datasets procesados.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd


def community_leiden(graph: nx.Graph | nx.DiGraph, seed: int = 42) -> list[set]:
    """! Detecta comunidades con el algoritmo de Leiden.

    Convierte el grafo de NetworkX a igraph, ejecuta Leiden con pesos
    y devuelve la partición como lista de sets de nodos (misma interfaz
    que `nx.community.louvain_communities`).

    Para grafos dirigidos se usa la versión no dirigida subyacente.

    @param graph Grafo de NetworkX (dirigido o no).
    @param seed  Semilla aleatoria para reproducibilidad.
    @return Lista de sets de nodos, uno por comunidad, ordenados por tamaño desc.
    """
    G_und = graph.to_undirected() if graph.is_directed() else graph
    if G_und.number_of_nodes() == 0:
        return []

    nodes = list(G_und.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    edges = [(node_index[u], node_index[v]) for u, v in G_und.edges()]
    weights = [G_und[u][v].get("weight", 1.0) for u, v in G_und.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
    )

    communities = [{nodes[i] for i in community} for community in partition]
    return sorted(communities, key=lambda c: -len(c))


def bigrams_for_tema(df: pd.DataFrame) -> pd.DataFrame:
    """! Genera bigramas por usuario (con conteos) para un tema.

    @param df DataFrame con columnas [user_id, pos, type].
    @return DataFrame con columnas [user_id, type_1, type_2, count].
    """
    if df.empty:
        return pd.DataFrame(columns=["type_1", "type_2", "count"])

    df = df[["user_id", "pos", "type"]].sort_values(["user_id", "pos"]).copy()
    df["type_2"] = df.groupby("user_id")["type"].shift(-1)
    df = df.dropna(subset=["type_2"])

    if df.empty:
        return pd.DataFrame(columns=["type_1", "type_2", "count"])

    bigrams = (
        df.groupby(["type", "type_2"])
        .size()
        .reset_index(name="count")
        .rename(columns={"type": "type_1"})
    )
    return bigrams


def bigrams_to_unordered(df_bigrams: pd.DataFrame) -> pd.DataFrame:
    """! Convierte bigramas ordenados a bigramas no ordenados.

    @param df_bigrams DataFrame con columnas [type_1, type_2, count].
    @return DataFrame con columnas [type_1, type_2, count] sin orden.
    """
    if df_bigrams.empty:
        return pd.DataFrame(columns=["type_1", "type_2", "count"])

    df = df_bigrams[["type_1", "type_2", "count"]].copy()
    types = df[["type_1", "type_2"]].astype(str).to_numpy()
    types_sorted = pd.DataFrame([sorted(pair) for pair in types], columns=["type_1", "type_2"])
    df[["type_1", "type_2"]] = types_sorted

    unordered = df.groupby(["type_1", "type_2"], dropna=False)["count"].sum().reset_index()
    return unordered


def bigrams_to_dirgraph(df_bigrams: pd.DataFrame) -> nx.DiGraph:
    """! Convierte un dataset de bigramas en un grafo dirigido con pesos.

    @param df_bigrams DataFrame con columnas [type_1, type_2, count].
    @return Graph grafo dirigido con pesos a partir de los bigramas.
    """

    G = nx.DiGraph()

    for _, row in df_bigrams.iterrows():
        G.add_node(row["type_1"])
        G.add_node(row["type_2"])
        G.add_weighted_edges_from([tuple(row.to_numpy())])

    return G


def bigrams_to_undgraph(df_bigrams: pd.DataFrame) -> nx.Graph:
    """! Convierte un dataset de bigramas en un grafo no dirigido con pesos.

    @param df_bigrams DataFrame con columnas [type_1, type_2, count].
    @return Graph grafo a partir de los bigramas.
    """

    G = nx.Graph()

    und_bigrams = bigrams_to_unordered(df_bigrams)

    for _, row in und_bigrams.iterrows():
        G.add_node(row["type_1"])
        G.add_node(row["type_2"])
        G.add_weighted_edges_from([tuple(row.to_numpy())])

    return G


def node_stats(graph: nx.Graph | nx.DiGraph) -> pd.DataFrame:
    """! Calcula estadísticas básicas por nodo.

    Columnas base:
    - node: etiqueta del nodo (type).
    - degree: grado no ponderado (número de vecinos/aristas).
    - degree_centrality: centralidad de grado (normalizada por NetworkX).
    - strength: grado ponderado usando `weight` como peso.
    - betweenness: betweenness centrality. intermediación no normalizada (`normalized=False`).
    - closeness: closeness centrality. cercanía con mejora de Wasserman-Faust (`wf_improved=True`).
    - pagerank: PageRank ponderado (`weight="weight"`, `alpha=0.85`, `tol=1e-6`).
    - eigenvector: centralidad eigenvector ponderada (`weight="weight"`, `max_iter=1000`).
    - clustering: coef. de clustering ponderado (`weight="weight"`).
    - strong_component_id: ID de componente fuertemente conexa (1..N).
    - weak_component_id: ID de componente débilmente conexa (1..N).

    Columnas extra si el grafo es dirigido:
    - in_degree / out_degree: grado entrante/saliente no ponderado.
    - in_strength / out_strength: grado entrante/saliente ponderado.

    @param graph Grafo de NetworkX (dirigido o no).
    @return DataFrame con estadísticas por nodo.
    """
    if graph.number_of_nodes() == 0:
        columns = [
            "node",
            "degree",
            "strength",
            "betweenness",
            "closeness",
            "pagerank",
            "strong_component_id",
            "weak_component_id",
        ]
        if graph.is_directed():
            columns.extend(["in_degree", "out_degree", "in_strength", "out_strength"])
        return pd.DataFrame(columns=columns)

    nodes = list(graph.nodes())

    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    strength = dict(graph.degree(weight="weight"))

    betweenness = nx.betweenness_centrality(graph, normalized=False)
    closeness = nx.closeness_centrality(graph, wf_improved=True)
    pagerank = nx.pagerank(graph, weight="weight", alpha=0.85, tol=1e-6)
    eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    clustering = nx.clustering(graph, weight="weight")

    def _ordered_components(components) -> list[set]:
        return sorted(components, key=lambda comp: (-len(comp), min(str(n) for n in comp)))

    if graph.is_directed():
        strong_components = _ordered_components(nx.strongly_connected_components(graph))
        weak_components = _ordered_components(nx.weakly_connected_components(graph))
    else:
        connected_components = _ordered_components(nx.connected_components(graph))
        strong_components = connected_components
        weak_components = connected_components

    strong_component_id = {
        node: comp_id
        for comp_id, component_nodes in enumerate(strong_components, start=1)
        for node in component_nodes
    }
    weak_component_id = {
        node: comp_id
        for comp_id, component_nodes in enumerate(weak_components, start=1)
        for node in component_nodes
    }

    data = {
        "node": nodes,
        "degree": [degree.get(n, 0) for n in nodes],
        "degree_centrality": [degree_centrality.get(n, 0.0) for n in nodes],
        "strength": [strength.get(n, 0.0) for n in nodes],
        "betweenness": [betweenness.get(n, 0.0) for n in nodes],
        "closeness": [closeness.get(n, 0.0) for n in nodes],
        "pagerank": [pagerank.get(n, 0.0) for n in nodes],
        "eigenvector": [eigenvector.get(n, 0.0) for n in nodes],
        "clustering": [clustering.get(n, 0.0) for n in nodes],
        "strong_component_id": [strong_component_id.get(n, 0) for n in nodes],
        "weak_component_id": [weak_component_id.get(n, 0) for n in nodes],
    }

    if graph.is_directed():
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        in_strength = dict(graph.in_degree(weight="weight"))
        out_strength = dict(graph.out_degree(weight="weight"))
        data.update(
            {
                "in_degree": [in_degree.get(n, 0) for n in nodes],
                "out_degree": [out_degree.get(n, 0) for n in nodes],
                "in_strength": [in_strength.get(n, 0.0) for n in nodes],
                "out_strength": [out_strength.get(n, 0.0) for n in nodes],
            }
        )

    out = pd.DataFrame(data)
    out = out.sort_values(["strength", "degree", "node"], ascending=[False, False, True])
    return out


ProgressCb = Callable[[int, int], None]  # (i_actual, n_total)
StatusCb = Callable[[str], None]


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def small_world_indices(
    graph: nx.Graph | nx.DiGraph,
    *,
    n: int = 10,
    progress_cb: Optional[ProgressCb] = None,
    status_cb: Optional[StatusCb] = None,
) -> dict:
    """! Calcula índices de small-world para grafos no dirigidos.

    @param graph Grafo de NetworkX (dirigido o no).
    @return Diccionario con SWI y ω junto a sus errores por métrica cuando no apliquen.
    """
    if graph.is_directed() or graph.number_of_nodes() <= 1:
        return {}

    L = nx.average_shortest_path_length(graph)
    C = nx.average_clustering(graph)

    Cl = -1
    Cr = -1
    Ll = 0.0
    Lr = 0.0
    start_time = perf_counter()

    for i in range(n):
        if progress_cb:
            progress_cb(i, n)
        if status_cb:
            if i == 0:
                status_cb(f"Iteración {i + 1}/{n}. ETA estimado: calculando...")
            else:
                elapsed = perf_counter() - start_time
                avg_per_iteration = elapsed / i
                eta = avg_per_iteration * (n - i)
                status_cb(f"Iteración {i + 1}/{n}. ETA estimado: {_format_eta(eta)}")

        lat = nx.algorithms.smallworld.lattice_reference(graph, niter=5, seed=42)
        rand = nx.algorithms.smallworld.random_reference(graph, niter=10, seed=42)

        Cl = max(Cl, nx.average_clustering(lat))
        Cr = max(Cr, nx.average_clustering(rand))
        Ll += nx.average_shortest_path_length(lat) / n
        Lr += nx.average_shortest_path_length(rand) / n

    # marca 100% al final
    if progress_cb:
        progress_cb(n, n)
    if status_cb:
        status_cb("Calculando ω y SWI...")

    result = {"SWI": None, "SWI_error": None, "ω": None, "ω_error": None}

    if L == 0:
        result["ω_error"] = "No se puede calcular ω: la longitud media del grafo es 0."
    elif Cl <= 0:
        result["ω_error"] = (
            "No se puede calcular ω: el clustering de referencia en red regular es 0."
        )
    else:
        result["ω"] = (Lr / L) - (C / Cl)

    if Ll == 0:
        result["SWI_error"] = "No se puede calcular SWI: la longitud media de la red regular es 0."
    elif (Lr - Ll) == 0:
        result["SWI_error"] = (
            "No se puede calcular SWI: la diferencia entre longitudes de referencia es 0."
        )
    elif (Cl - Cr) == 0:
        result["SWI_error"] = (
            "No se puede calcular SWI: la diferencia entre clustering de referencia es 0."
        )
    else:
        result["SWI"] = ((L / Ll) / (Lr - Ll)) * ((C - Cr) / (Cl - Cr))

    if status_cb:
        status_cb("Listo ✅")

    return result


def is_graph_connected(graph: nx.Graph | nx.DiGraph) -> bool:
    """! Comprueba si el grafo es conexo.

    Para grafos dirigidos evalúa conectividad débil.
    """
    if graph.number_of_nodes() == 0:
        return False
    if graph.is_directed():
        return nx.is_weakly_connected(graph)
    return nx.is_connected(graph)


def connected_components_sorted(graph: nx.Graph | nx.DiGraph) -> list[set]:
    """! Devuelve componentes ordenadas por tamaño desc y etiqueta mínima asc."""
    if graph.number_of_nodes() == 0:
        return []
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    return sorted(components, key=lambda comp: (-len(comp), min(str(n) for n in comp)))


def graph_stats(
    graph: nx.Graph | nx.DiGraph, node_stats_df: pd.DataFrame, include_small_world: bool = False
) -> dict:
    """! Calcula estadísticas generales del grafo.

    Métricas:
    - diameter: diámetro de la mayor componente fuertemente conexa (dirigido) o conexa (no dirigido).
    - avg_clustering: coeficiente de agrupamiento promedio (`nx.average_clustering`, ponderado).
    - avg_degree: grado promedio (a partir de `node_stats_df["degree"]`).
    - avg_strength: fuerza promedio (a partir de `node_stats_df["strength"]`).
    - density: densidad del grafo (`nx.density`).
    - components: número de componentes (fuertemente conexas si es dirigido, conexas si no).
    - avg_path_length: longitud de camino promedio sin pesos en la mayor componente conexa.
    - assortativity: coeficiente de asortatividad de grado (`nx.degree_assortativity_coefficient`). `None` si no se puede calcular.

    Si es no dirigido (y `include_small_world=True`):
    - SWI: Small-worldness index.
    - ω: omega de small world

    @param graph Grafo de NetworkX (dirigido o no).
    @param node_stats_df DataFrame de estadísticas por nodo (salida de `node_stats`).
    @param include_small_world Si `True`, añade SWI y ω para grafos no dirigidos.
    @return Diccionario con métricas generales del grafo.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return {
            "diameter": 0,
            "avg_clustering": 0.0,
            "avg_degree": 0.0,
            "avg_strength": 0.0,
            "density": 0.0,
            "components": 0,
            "avg_path_length": 0.0,
            "assortativity": None,
        }

    if graph.is_directed():
        sccomponents = list(nx.strongly_connected_components(graph))
        wccomponents = list(nx.weakly_connected_components(graph))
        largest_nodes = max(sccomponents, key=len)
        largest = graph.subgraph(largest_nodes).copy()
    else:
        sccomponents = list(nx.connected_components(graph))
        wccomponents = sccomponents
        largest_nodes = max(sccomponents, key=len)
        largest = graph.subgraph(largest_nodes).copy()

    if largest.number_of_nodes() <= 1:
        diameter = 0
        avg_path_length = 0.0
    else:
        diameter = nx.diameter(largest)
        avg_path_length = nx.average_shortest_path_length(largest)

    avg_degree = float(node_stats_df["degree"].mean() / 2) if "degree" in node_stats_df else 0.0
    avg_strength = (
        float(node_stats_df["strength"].mean() / 2) if "strength" in node_stats_df else 0.0
    )

    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    except (nx.NetworkXError, ZeroDivisionError):
        assortativity = None

    ret = {
        "diameter": diameter,
        "avg_clustering": float(nx.average_clustering(graph, weight="weight")),
        "avg_degree": avg_degree,
        "avg_strength": avg_strength,
        "density": float(nx.density(graph)),
        "components": len(wccomponents),
        "avg_path_length": avg_path_length,
        "assortativity": assortativity,
    }

    if include_small_world:
        ret.update(small_world_indices(graph))

    return ret
