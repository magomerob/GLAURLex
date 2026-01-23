import numpy as np
import pandas as pd


def estadisticas_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas: token (str), pos (int), user_id (int/str)
    Devuelve: token, freq_rel, disponibilidad, avg_pos, aparicion, freq_acum
    """

    if df.empty:
        return pd.DataFrame(
            columns=["token", "disponibilidad", "avg_pos", "aparición", "freq_rel", "freq_acum"]
        )

    ninf = df["user_id"].nunique()

    # Pos máxima alcanzada
    n = int(df["pos"].max()) if pd.notna(df["pos"].max()) else 0
    if n == 0:
        # Caso límite: todo pos=0
        out = (
            df["token"]
            .value_counts(normalize=True)
            .rename("freq_rel")
            .reset_index()
            .rename(columns={"index": "token"})
        )
        out["disponibilidad"] = 1.0
        out["avg_pos"] = 0.0
        out["aparición"] = (
            df.groupby("token")["user_id"].nunique().reindex(out["token"]).to_numpy() / ninf
        )
        out = out.sort_values("disponibilidad", ascending=False)
        out["freq_acum"] = out["freq_rel"].cumsum()
        out = out.rename(columns={"aparición": "aparición"})
        return out

    # Frecuencia relativa por token
    freq = (
        df["token"]
        .value_counts(normalize=True)
        .rename("freq_rel")
        .reset_index()
        .rename(columns={"index": "token"})
    )

    # avg_pos por token
    avg_pos = df.groupby("token", as_index=False)["pos"].mean().rename(columns={"pos": "avg_pos"})

    # aparición = nº informantes que lo dijeron / ninf
    apar = df.groupby("token")["user_id"].nunique().rename("aparición").reset_index()
    apar["aparición"] = apar["aparición"] / ninf

    # disponibilidad (vectorizada):
    # 1) contar informantes únicos por (token, pos)
    tp = df.groupby(["token", "pos"])["user_id"].nunique().rename("u").reset_index()

    # 2) pesos por pos
    # exp(-2.3 * (i/(n)))
    weights = np.exp(-2.3 * (tp["pos"].to_numpy() / (n)))

    # 3) sumar e * (u/ninf) por token
    tp["w"] = weights
    tp["term"] = tp["w"] * (tp["u"] / ninf)
    disp = (
        tp.groupby("token", as_index=False)["term"].sum().rename(columns={"term": "disponibilidad"})
    )

    # juntar todo
    out = (
        freq.merge(disp, on="token", how="left")
        .merge(avg_pos, on="token", how="left")
        .merge(apar, on="token", how="left")
    )

    out["disponibilidad"] = out["disponibilidad"].fillna(0.0)
    out["avg_pos"] = out["avg_pos"].fillna(np.nan)
    out["aparición"] = out["aparición"].fillna(0.0)

    out = out.sort_values(by=["disponibilidad"], ascending=False)
    out["freq_acum"] = out["freq_rel"].cumsum()

    return out


def estadisticas(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return estadisticas_df(df)
