"""! @package urlex.core.graph
Utilidades de grafos para datasets procesados.
"""

from __future__ import annotations

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
    tokens_sorted = pd.DataFrame(
        [sorted(pair) for pair in tokens], columns=["token_1", "token_2"]
    )
    df[["token_1", "token_2"]] = tokens_sorted

    unordered = (
        df.groupby(["token_1", "token_2"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    return unordered
