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

    Columnas extra si el grafo es dirigido:
    - in_degree / out_degree: grado entrante/saliente no ponderado.
    - in_strength / out_strength: grado entrante/saliente ponderado.

    @param graph Grafo de NetworkX (dirigido o no).
    @return DataFrame con estadísticas por nodo.
    """
    if graph.number_of_nodes() == 0:
        columns = ["node", "degree", "strength", "betweenness", "closeness", "pagerank"]
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
