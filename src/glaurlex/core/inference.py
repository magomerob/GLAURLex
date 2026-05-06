"""! @package glaurlex.core.inference
Análisis estadístico descriptivo e inferencial sobre métricas léxicas
y variables sociolinguísticas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from glaurlex.core.stats import estadisticas_df


# ---------------------------------------------------------------------------
# Métricas por informante
# ---------------------------------------------------------------------------


def _shannon_entropy(counts: np.ndarray) -> float:
    """! Entropía de Shannon (base 2) sobre un vector de conteos."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def informant_metrics(
    df_tema: pd.DataFrame,
    informantes: pd.DataFrame,
    type_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """! Calcula métricas léxicas por informante a partir de un tema.

    Cada fila corresponde a un informante presente en `df_tema`. Se mergea con
    la tabla de informantes para enriquecer con las variables sociolinguísticas.

    Métricas devueltas:
    - n_tokens: número total de respuestas (tokens) producidas.
    - n_types: número de types distintos producidos.
    - ttr: Type-Token Ratio (n_types / n_tokens).
    - mean_pos: posición media (1-indexed) de las respuestas.
    - max_pos: última posición alcanzada (longitud de la lista).
    - entropy: entropía de Shannon (bits) sobre la distribución de types.
    - total_disp: suma de disponibilidad de los types producidos
      (basada en las estadísticas de tema en el grupo activo).
    - mean_disp: media de disponibilidad de los types producidos.

    @param df_tema DataFrame con columnas [user_id, pos, type] (ya filtrado por grupo).
    @param informantes DataFrame de informantes (con sus variables sociolinguísticas).
    @param type_stats DataFrame opcional con estadísticas por type (salida de
        `estadisticas_df`); si no se pasa, se calcula sobre `df_tema`.
    @return DataFrame con una fila por informante y las métricas + variables.
    """
    if df_tema.empty:
        return pd.DataFrame()

    if type_stats is None:
        type_stats = estadisticas_df(df_tema)

    disp_map = (
        type_stats.set_index("type")["disponibilidad"].to_dict()
        if "disponibilidad" in type_stats.columns
        else {}
    )

    grouped = df_tema.groupby("user_id")
    rows = []
    for uid, sub in grouped:
        types_arr = sub["type"].to_numpy()
        pos_arr = sub["pos"].to_numpy()
        counts = sub["type"].value_counts().to_numpy()
        disps = np.array([disp_map.get(t, 0.0) for t in types_arr], dtype=float)

        n_tok = int(len(types_arr))
        n_typ = int(sub["type"].nunique())
        rows.append(
            {
                "user_id": uid,
                "n_tokens": n_tok,
                "n_types": n_typ,
                "ttr": (n_typ / n_tok) if n_tok > 0 else 0.0,
                "mean_pos": float(pos_arr.mean()) + 1 if n_tok > 0 else np.nan,
                "max_pos": int(pos_arr.max()) + 1 if n_tok > 0 else 0,
                "entropy": _shannon_entropy(counts),
                "total_disp": float(disps.sum()),
                "mean_disp": float(disps.mean()) if n_tok > 0 else 0.0,
            }
        )

    metrics_df = pd.DataFrame(rows)

    # Mergear con informantes para incluir las variables sociolinguísticas
    inf = informantes.copy()
    if "CODIGO_INFORMANTE" in inf.columns:
        merged = metrics_df.merge(
            inf, left_on="user_id", right_on="CODIGO_INFORMANTE", how="left"
        )
    else:
        inf = inf.reset_index(drop=False).rename(columns={"index": "_inf_idx"})
        inf["user_id"] = inf["_inf_idx"] + 1
        inf = inf.drop(columns=["_inf_idx"])
        merged = metrics_df.merge(inf, on="user_id", how="left")

    return merged


# ---------------------------------------------------------------------------
# Análisis descriptivo
# ---------------------------------------------------------------------------


def describe_series(s: pd.Series) -> Dict[str, Any]:
    """! Estadísticos descriptivos detallados sobre una serie numérica.

    Devuelve N, media, std, error estándar, mínimo, percentiles 5/25/50/75/95,
    máximo, IQR, rango, varianza, coeficiente de variación, asimetría y curtosis.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return {"N": 0}

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    return {
        "N": n,
        "media": mean,
        "std": std,
        "se": std / np.sqrt(n) if n > 0 else np.nan,
        "min": float(s.min()),
        "p5": float(np.percentile(s, 5)),
        "Q1": float(np.percentile(s, 25)),
        "mediana": float(np.percentile(s, 50)),
        "Q3": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "max": float(s.max()),
        "IQR": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "rango": float(s.max() - s.min()),
        "varianza": float(s.var(ddof=1)) if n > 1 else 0.0,
        "CV": (std / mean) if mean != 0 else np.nan,
        "asimetría": float(stats.skew(s, bias=False)) if n > 2 else np.nan,
        "curtosis": float(stats.kurtosis(s, bias=False)) if n > 3 else np.nan,
    }


def normality_test(s: pd.Series) -> Dict[str, Any]:
    """! Test de normalidad de Shapiro-Wilk.

    @return Diccionario con W, p_value, n y conclusión heurística (α=0.05).
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s.shape[0])
    if n < 3:
        return {"test": "Shapiro-Wilk", "n": n, "error": "N < 3"}
    if n > 5000:
        # Shapiro-Wilk no es fiable con N muy grande; usar D'Agostino
        k2, p = stats.normaltest(s)
        return {
            "test": "D'Agostino-Pearson",
            "n": n,
            "estadístico": float(k2),
            "p_value": float(p),
            "normal_α=0.05": bool(p > 0.05),
        }
    w, p = stats.shapiro(s)
    return {
        "test": "Shapiro-Wilk",
        "n": n,
        "estadístico": float(w),
        "p_value": float(p),
        "normal_α=0.05": bool(p > 0.05),
    }


# ---------------------------------------------------------------------------
# Inferencia: comparación entre grupos
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """! d de Cohen con varianza agrupada (pooled)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    """! Tamaño del efecto rank-biserial a partir del estadístico U."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * u) / (n1 * n2))


def _eta_squared_oneway(groups: List[np.ndarray]) -> float:
    """! η² para ANOVA one-way (proporción de varianza explicada)."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
    ss_total = float(((all_vals - grand_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def _epsilon_squared_kw(h: float, n: int, k: int) -> float:
    """! ε² para Kruskal-Wallis (tamaño de efecto)."""
    if n <= k:
        return float("nan")
    return float((h - k + 1) / (n - k))


@dataclass
class InferenceResult:
    """! Resultado estructurado de un análisis inferencial."""

    metric: str
    by: str
    kind: str  # "two-group", "k-group", "correlation"
    n_total: int
    descriptives: pd.DataFrame
    parametric: Dict[str, Any] = field(default_factory=dict)
    non_parametric: Dict[str, Any] = field(default_factory=dict)
    effect_size: Dict[str, Any] = field(default_factory=dict)
    posthoc: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)


def _group_descriptives(df: pd.DataFrame, metric: str, by: str) -> pd.DataFrame:
    rows = []
    for level, sub in df.groupby(by, dropna=True):
        s = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                by: level,
                "N": int(s.shape[0]),
                "media": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "mediana": float(s.median()),
                "Q1": float(np.percentile(s, 25)),
                "Q3": float(np.percentile(s, 75)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by).reset_index(drop=True)
    return out


def compare_groups(
    df: pd.DataFrame,
    metric: str,
    by: str,
    *,
    posthoc: bool = True,
    posthoc_alpha: float = 0.05,
) -> InferenceResult:
    """! Compara una métrica numérica a través de los niveles de una variable.

    Selecciona automáticamente:
    - 2 grupos: t de Welch + Mann-Whitney U + Cohen's d + rank-biserial.
    - 3+ grupos: ANOVA one-way + Kruskal-Wallis + η² + ε² (+ post-hoc Mann-Whitney
      pareados con corrección de Bonferroni si `posthoc=True`).

    @param df DataFrame con columnas `metric` y `by`.
    @param metric Nombre de la columna numérica a comparar.
    @param by Nombre de la columna categórica.
    @return InferenceResult con descriptivos, tests y tamaños de efecto.
    """
    work = df[[metric, by]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric, by])

    levels = sorted(work[by].dropna().unique().tolist(), key=str)
    groups = [work.loc[work[by] == lv, metric].to_numpy() for lv in levels]
    groups = [g for g in groups if len(g) >= 2]

    desc = _group_descriptives(work, metric, by)
    notes: List[str] = []

    if len(groups) < 2:
        return InferenceResult(
            metric=metric,
            by=by,
            kind="insufficient",
            n_total=int(len(work)),
            descriptives=desc,
            notes=["Se necesitan ≥2 niveles con N≥2 cada uno."],
        )

    n_total = int(sum(len(g) for g in groups))

    if len(groups) == 2:
        a, b = groups
        try:
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        except Exception as exc:  # pragma: no cover
            t_stat, t_p = float("nan"), float("nan")
            notes.append(f"t-test: {exc}")
        try:
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception as exc:
            u_stat, u_p = float("nan"), float("nan")
            notes.append(f"Mann-Whitney: {exc}")

        return InferenceResult(
            metric=metric,
            by=by,
            kind="two-group",
            n_total=n_total,
            descriptives=desc,
            parametric={
                "test": "t de Welch (independientes, var. desiguales)",
                "estadístico": float(t_stat),
                "p_value": float(t_p),
                "gl_aprox": float(len(a) + len(b) - 2),
            },
            non_parametric={
                "test": "Mann-Whitney U (dos colas)",
                "U": float(u_stat),
                "p_value": float(u_p),
            },
            effect_size={
                "Cohen's d": _cohens_d(a, b),
                "rank-biserial r": _rank_biserial(float(u_stat), len(a), len(b)),
            },
            notes=notes,
        )

    # k-group
    try:
        f_stat, f_p = stats.f_oneway(*groups)
    except Exception as exc:
        f_stat, f_p = float("nan"), float("nan")
        notes.append(f"ANOVA: {exc}")
    try:
        h_stat, h_p = stats.kruskal(*groups)
    except Exception as exc:
        h_stat, h_p = float("nan"), float("nan")
        notes.append(f"Kruskal-Wallis: {exc}")

    eta2 = _eta_squared_oneway(groups)
    eps2 = _epsilon_squared_kw(float(h_stat), n_total, len(groups))

    posthoc_df = None
    if posthoc and len(groups) >= 3:
        rows = []
        levels_used = [lv for lv, g in zip(levels, [groups[i] for i in range(len(groups))]) if len(g) >= 2]
        m = len(levels_used) * (len(levels_used) - 1) // 2
        for i in range(len(levels_used)):
            for j in range(i + 1, len(levels_used)):
                a = work.loc[work[by] == levels_used[i], metric].to_numpy()
                b = work.loc[work[by] == levels_used[j], metric].to_numpy()
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    u, p = float("nan"), float("nan")
                p_adj = min(p * m, 1.0) if not np.isnan(p) else float("nan")
                rows.append(
                    {
                        "grupo_1": levels_used[i],
                        "grupo_2": levels_used[j],
                        "n_1": int(len(a)),
                        "n_2": int(len(b)),
                        "U": float(u),
                        "p_value": float(p),
                        "p_Bonferroni": float(p_adj),
                        "significativo_α": bool(p_adj < posthoc_alpha),
                        "rank-biserial r": _rank_biserial(float(u), len(a), len(b)),
                    }
                )
        posthoc_df = pd.DataFrame(rows)

    return InferenceResult(
        metric=metric,
        by=by,
        kind="k-group",
        n_total=n_total,
        descriptives=desc,
        parametric={
            "test": "ANOVA one-way",
            "F": float(f_stat),
            "p_value": float(f_p),
            "gl_entre": float(len(groups) - 1),
            "gl_dentro": float(n_total - len(groups)),
        },
        non_parametric={
            "test": "Kruskal-Wallis H",
            "H": float(h_stat),
            "p_value": float(h_p),
            "gl": float(len(groups) - 1),
        },
        effect_size={
            "η² (ANOVA)": eta2,
            "ε² (Kruskal-Wallis)": eps2,
        },
        posthoc=posthoc_df,
        notes=notes,
    )


def correlation(
    df: pd.DataFrame,
    metric: str,
    var: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """! Correlación de Pearson y Spearman entre dos columnas numéricas.

    @return (resultado, datos_limpios) con r, p y N para cada test.
    """
    work = df[[metric, var]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work[var] = pd.to_numeric(work[var], errors="coerce")
    work = work.dropna()
    n = int(work.shape[0])
    if n < 3:
        return ({"N": n, "error": "Se necesitan ≥3 observaciones."}, work)

    r_p, p_p = stats.pearsonr(work[metric], work[var])
    r_s, p_s = stats.spearmanr(work[metric], work[var])
    # regresión lineal simple
    slope, intercept, r_lin, p_lin, se_lin = stats.linregress(work[var], work[metric])
    return (
        {
            "N": n,
            "Pearson r": float(r_p),
            "Pearson p_value": float(p_p),
            "Spearman ρ": float(r_s),
            "Spearman p_value": float(p_s),
            "regresión_pendiente": float(slope),
            "regresión_intercepto": float(intercept),
            "regresión_se": float(se_lin),
            "regresión_p_value": float(p_lin),
            "R²": float(r_lin ** 2),
        },
        work,
    )
