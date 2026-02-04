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
