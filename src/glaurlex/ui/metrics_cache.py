"""! @package glaurlex.ui.metrics_cache
Cálculo cacheado y compartido de métricas léxicas y de grafo para las vistas.

Centraliza los wrappers `@st.cache_data` para evitar que distintas vistas
recomputen las mismas métricas (estadísticas por type, métricas por informante,
métricas de nodo, bigramas).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from glaurlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    node_stats,
)
from glaurlex.core.inference import informant_metrics
from glaurlex.core.stats import estadisticas_df


_INFORMANT_COL_CANDIDATES = (
    "CODIGO_INFORMANTE",
    "codigoinformante",
    "codigo_informante",
    "informante",
    "user",
    "usuario",
    "center",
    "centers",
    "user_id",
)


def infer_informant_col(df_tema: pd.DataFrame) -> str | None:
    """! Devuelve el nombre de la columna que identifica al informante."""
    return next((c for c in _INFORMANT_COL_CANDIDATES if c in df_tema.columns), None)


def filter_by_group(
    df_tema: pd.DataFrame,
    informantes_f: pd.DataFrame | None,
    informant_col: str | None,
) -> pd.DataFrame:
    """! Filtra `df_tema` para conservar solo los informantes del grupo activo."""
    if informantes_f is None or informant_col is None:
        return df_tema
    id_col = "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
    allowed = set(
        (informantes_f.index + 1).tolist() if id_col is None else informantes_f[id_col].tolist()
    )
    return df_tema[df_tema[informant_col].isin(allowed)]


@st.cache_data(show_spinner=False)
def type_stats_cached(df_tema: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    """! Estadísticas por type (cacheado)."""
    _ = cache_key
    return estadisticas_df(df_tema)


@st.cache_data(show_spinner=False)
def bigrams_cached(df_tema: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    """! Bigramas por tema (cacheado)."""
    _ = cache_key
    return bigrams_for_tema(df_tema)


@st.cache_data(show_spinner=False)
def node_stats_cached(df_tema: pd.DataFrame, directed: bool, cache_key: str) -> pd.DataFrame:
    """! Estadísticas por nodo del grafo (cacheado).

    Reutiliza `bigrams_cached` para evitar recomputar bigramas.
    """
    bigrams = bigrams_cached(df_tema, cache_key)
    G = bigrams_to_dirgraph(bigrams) if directed else bigrams_to_undgraph(bigrams)
    return node_stats(G)


@st.cache_data(show_spinner=False)
def informant_metrics_cached(
    df_tema: pd.DataFrame,
    informantes: pd.DataFrame,
    cache_key: str,
) -> pd.DataFrame:
    """! Métricas por informante (cacheado).

    Reutiliza `type_stats_cached` para no recomputar las estadísticas por type.
    """
    type_stats = type_stats_cached(df_tema, cache_key)
    return informant_metrics(df_tema, informantes, type_stats=type_stats)
