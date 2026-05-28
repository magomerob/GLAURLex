import numpy as np
import pandas as pd

"""! @package glaurlex.core.stats
Funciones estadísticas sobre datasets de tipos léxicos.
"""


def estadisticas_df(df: pd.DataFrame, informantes_df: pd.DataFrame = None) -> pd.DataFrame:
    """! Calcula estadísticas básicas para types.

    Espera columnas: type (str), pos (int), user_id (int/str).
    Devuelve columnas: type, tokens, freq_rel, disponibilidad, avg_pos, aparición, freq_acum.

    @param df DataFrame con ocurrencias de types.
    @param informantes_df DataFrame con informantes (para filtrar user_id).
    @return DataFrame con estadísticas por type.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "type",
                "tokens",
                "disponibilidad",
                "avg_pos",
                "aparición",
                "freq_rel",
                "freq_acum",
            ]
        )

    if informantes_df is not None:
        allowed = set(informantes_df["CODIGO_INFORMANTE"].tolist())
        df = df[df["user_id"].isin(allowed)]

    ninf = df["user_id"].nunique()

    # Pos máxima alcanzada
    n = int(df["pos"].max()) if pd.notna(df["pos"].max()) else 0
    if n == 0:
        # Caso límite: todo pos=0
        counts = (
            df["type"]
            .value_counts()
            .rename("tokens")
            .reset_index()
            .rename(columns={"index": "type"})
        )
        out = (
            df["type"]
            .value_counts(normalize=True)
            .rename("freq_rel")
            .reset_index()
            .rename(columns={"index": "type"})
        )
        out = out.merge(counts, on="type", how="left")
        out["disponibilidad"] = 1.0
        out["avg_pos"] = 0.0
        out["aparición"] = (
            df.groupby("type")["user_id"].nunique().reindex(out["type"]).to_numpy() / ninf
        )
        out = out.sort_values("disponibilidad", ascending=False)
        out["freq_acum"] = out["freq_rel"].cumsum()
        return out

    # Conteo de tokens por type (número de ocurrencias)
    counts = (
        df["type"].value_counts().rename("tokens").reset_index().rename(columns={"index": "type"})
    )

    # Frecuencia relativa por type
    freq = (
        df["type"]
        .value_counts(normalize=True)
        .rename("freq_rel")
        .reset_index()
        .rename(columns={"index": "type"})
    )

    # avg_pos por type
    avg_pos = df.groupby("type", as_index=False)["pos"].mean().rename(columns={"pos": "avg_pos"})

    # aparición = nº informantes que lo dijeron / ninf
    apar = df.groupby("type")["user_id"].nunique().rename("aparición").reset_index()
    apar["aparición"] = apar["aparición"] / ninf

    # disponibilidad (vectorizada):
    # 1) contar informantes únicos por (type, pos)
    tp = df.groupby(["type", "pos"])["user_id"].nunique().rename("u").reset_index()

    # 2) pesos por pos
    # exp(-2.3 * (i/(n)))
    weights = np.exp(-2.3 * (tp["pos"].to_numpy() / (n)))

    # 3) sumar e * (u/ninf) por type
    tp["w"] = weights
    tp["term"] = tp["w"] * (tp["u"] / ninf)
    disp = (
        tp.groupby("type", as_index=False)["term"].sum().rename(columns={"term": "disponibilidad"})
    )

    # juntar todo
    out = (
        freq.merge(counts, on="type", how="left")
        .merge(disp, on="type", how="left")
        .merge(avg_pos, on="type", how="left")
        .merge(apar, on="type", how="left")
    )

    out["tokens"] = out["tokens"].fillna(0).astype(int)
    out["disponibilidad"] = out["disponibilidad"].fillna(0.0)
    out["avg_pos"] = out["avg_pos"].fillna(np.nan) + 1
    out["aparición"] = out["aparición"].fillna(0.0)

    out = out.sort_values(by=["disponibilidad"], ascending=False)
    out["freq_acum"] = out["freq_rel"].cumsum()

    return out


def estadisticas(path: str) -> pd.DataFrame:
    """! Carga un parquet y devuelve sus estadísticas.

    @param path Ruta al parquet.
    @return DataFrame con estadísticas por type.
    """
    df = pd.read_parquet(path)
    return estadisticas_df(df)
