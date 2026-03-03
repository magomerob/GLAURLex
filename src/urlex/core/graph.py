"""! @package urlex.core.graph
Utilidades de grafos para datasets procesados.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def bigrams_for_tema(df: pd.DataFrame) -> pd.DataFrame:
    """! Genera bigramas por usuario (con conteos) para un tema.

    @param df DataFrame con columnas [user_id, pos, token].
    @return DataFrame con columnas [user_id, token_1, token_2, count].
    """
    if df.empty:
        return pd.DataFrame(columns=["token_1", "token_2", "count"])

    df = df[["user_id", "pos", "token"]].sort_values(["user_id", "pos"]).copy()
    df["token_2"] = df.groupby("user_id")["token"].shift(-1)
    df = df.dropna(subset=["token_2"])

    if df.empty:
        return pd.DataFrame(columns=["token_1", "token_2", "count"])

    bigrams = (
        df.groupby(["token", "token_2"])
        .size()
        .reset_index(name="count")
        .rename(columns={"token": "token_1"})
    )
    return bigrams


def bigrams_to_unordered(df_bigrams: pd.DataFrame) -> pd.DataFrame:
    """! Convierte bigramas ordenados a bigramas no ordenados.

    @param df_bigrams DataFrame con columnas [token_1, token_2, count].
    @return DataFrame con columnas [token_1, token_2, count] sin orden.
    """
    if df_bigrams.empty:
        return pd.DataFrame(columns=["token_1", "token_2", "count"])

    df = df_bigrams[["token_1", "token_2", "count"]].copy()
    tokens = df[["token_1", "token_2"]].astype(str).to_numpy()
    tokens_sorted = pd.DataFrame([sorted(pair) for pair in tokens], columns=["token_1", "token_2"])
    df[["token_1", "token_2"]] = tokens_sorted

    unordered = df.groupby(["token_1", "token_2"], dropna=False)["count"].sum().reset_index()
    return unordered


def bigrams_to_dirgraph(df_bigrams: pd.DataFrame) -> nx.DiGraph:
    """! Convierte un dataset de bigramas en un grafo dirigido con pesos.

    @param df_bigrams DataFrame con columnas [token_1, token_2, count].
    @return Graph grafo dirigido con pesos a partir de los bigramas.
    """

    G = nx.DiGraph()

    for _, row in df_bigrams.iterrows():
        G.add_node(row["token_1"])
        G.add_node(row["token_2"])
        G.add_weighted_edges_from([tuple(row.to_numpy())])

    return G


def bigrams_to_undgraph(df_bigrams: pd.DataFrame) -> nx.Graph:
    """! Convierte un dataset de bigramas en un grafo no dirigido con pesos.

    @param df_bigrams DataFrame con columnas [token_1, token_2, count].
    @return Graph grafo a partir de los bigramas.
    """

    G = nx.Graph()

    und_bigrams = bigrams_to_unordered(df_bigrams)

    for _, row in und_bigrams.iterrows():
        G.add_node(row["token_1"])
        G.add_node(row["token_2"])
        G.add_weighted_edges_from([tuple(row.to_numpy())])

    return G


def node_stats(graph: nx.Graph | nx.DiGraph) -> pd.DataFrame:
    """! Calcula estadísticas básicas por nodo.

    Columnas base:
    - node: etiqueta del nodo (token).
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


def small_world_indices(graph: nx.Graph | nx.DiGraph) -> dict:
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
    n = 10
    for i in range(n):
        print(i, "/10")
        lat = nx.algorithms.smallworld.lattice_reference(graph, niter=5, seed=42)
        rand = nx.algorithms.smallworld.random_reference(graph, niter=10, seed=42)

        Cl = max(Cl, nx.average_clustering(lat))
        Cr = max(Cr, nx.average_clustering(rand))
        Ll += nx.average_shortest_path_length(lat) / n
        Lr += nx.average_shortest_path_length(rand) / n

    result = {
        "SWI": None,
        "SWI_error": None,
        "ω": None,
        "ω_error": None,
    }

    if L == 0:
        result["ω_error"] = "No se puede calcular ω: la longitud media del grafo es 0."
    elif Cl <= 0:
        result["ω_error"] = (
            "No se puede calcular ω: el clustering de referencia en red regular es 0."
        )
    else:
        w = (Lr / L) - (C / Cl)
        result["ω"] = w
        print("ω", result["ω"])

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
        print("SWI", result["SWI"])

    if result["SWI_error"] and result["ω_error"]:
        print("fallo en el cálculo")

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

    ret = {
        "diameter": diameter,
        "avg_clustering": float(nx.average_clustering(graph, weight="weight")),
        "avg_degree": avg_degree,
        "avg_strength": avg_strength,
        "density": float(nx.density(graph)),
        "components": len(wccomponents),
        "avg_path_length": avg_path_length,
    }

    if include_small_world:
        ret.update(small_world_indices(graph))

    return ret
