"""! @package glaurlex.ui.views.inference
Vista de análisis estadístico descriptivo e inferencial sobre métricas
léxicas y variables sociolinguísticas.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats as sp_stats

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    node_stats,
)
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.core.inference import (
    compare_groups,
    correlation,
    describe_series,
    informant_metrics,
    normality_test,
)
from glaurlex.core.stats import estadisticas_df
from glaurlex.ui.state import ensure_groups_loaded_for_dataset, ensure_state

# ---------------------------------------------------------------------------
# Catálogos de métricas
# ---------------------------------------------------------------------------

INFORMANT_METRICS: dict[str, str] = {
    "n_tokens": "Tokens producidos",
    "n_types": "Types distintos",
    "ttr": "Type-Token Ratio (TTR)",
    "mean_pos": "Posición media",
    "max_pos": "Longitud (max_pos)",
    "entropy": "Entropía de Shannon",
    "total_disp": "Disponibilidad acumulada",
    "mean_disp": "Disponibilidad media",
}

TYPE_METRICS: dict[str, str] = {
    "disponibilidad": "Disponibilidad",
    "aparición": "Aparición",
    "freq_rel": "Frecuencia relativa",
    "avg_pos": "Posición promedio",
    "freq_acum": "Frecuencia acumulada",
    "tokens": "Tokens",
}

NODE_METRICS: dict[str, str] = {
    "degree": "Grado",
    "degree_centrality": "Centralidad de grado",
    "strength": "Fuerza (grado ponderado)",
    "betweenness": "Intermediación",
    "closeness": "Cercanía",
    "pagerank": "PageRank",
    "eigenvector": "Eigenvector",
    "clustering": "Clustering",
}

# Columnas que NO sirven como variable sociolinguística
_INFORMANT_DROP_COLS = {
    "user_id",
    "CODIGO_INFORMANTE",
    "codigoinformante",
    "codigo_informante",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data
def _load_dataset(processed_dir: str, name: str):
    return _get_service(processed_dir).load_processed(name)


@st.cache_data(show_spinner=False)
def _type_stats_cached(df_tema: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    return estadisticas_df(df_tema)


@st.cache_data(show_spinner=False)
def _node_stats_cached(df_tema: pd.DataFrame, directed: bool, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    bigrams = bigrams_for_tema(df_tema)
    G = bigrams_to_dirgraph(bigrams) if directed else bigrams_to_undgraph(bigrams)
    return node_stats(G)


@st.cache_data(show_spinner=False)
def _informant_metrics_cached(
    df_tema: pd.DataFrame,
    informantes: pd.DataFrame,
    cache_key: str,
) -> pd.DataFrame:
    _ = cache_key
    type_stats = estadisticas_df(df_tema)
    return informant_metrics(df_tema, informantes, type_stats=type_stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_informant_col(df_tema: pd.DataFrame) -> Optional[str]:
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
    return next((c for c in candidates if c in df_tema.columns), None)


def _filter_by_group(
    df_tema: pd.DataFrame, informantes_f, informant_col: Optional[str]
) -> pd.DataFrame:
    if informantes_f is None or informant_col is None:
        return df_tema
    id_col = "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
    allowed = set(
        (informantes_f.index + 1).tolist() if id_col is None else informantes_f[id_col].tolist()
    )
    return df_tema[df_tema[informant_col].isin(allowed)]


def _is_numeric_high_card(series: pd.Series, min_unique: int = 8) -> bool:
    """! Heurística: ¿la serie debe tratarse como numérica continua?"""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().mean() < 0.9:
        return False
    return s.dropna().nunique() >= min_unique


def _categorize_variables(
    df: pd.DataFrame, candidates: Iterable[str]
) -> tuple[list[str], list[str]]:
    """! Separa columnas en (categóricas, numéricas continuas)."""
    cats: list[str] = []
    nums: list[str] = []
    for c in candidates:
        if c not in df.columns:
            continue
        if _is_numeric_high_card(df[c]):
            nums.append(c)
        else:
            cats.append(c)
    return sorted(cats), sorted(nums)


def _format_pvalue(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _significance_badge(p: float, alpha: float = 0.05) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "★★★ (p<0.001)"
    if p < 0.01:
        return "★★ (p<0.01)"
    if p < alpha:
        return "★ (p<0.05)"
    return "ns"


# ---------------------------------------------------------------------------
# Sub-vistas
# ---------------------------------------------------------------------------


def _render_descriptive_section(
    metric_values: pd.Series,
    metric_label: str,
    by_var: Optional[str] = None,
    df_full: Optional[pd.DataFrame] = None,
):
    s = pd.to_numeric(metric_values, errors="coerce").dropna()
    if s.empty:
        st.warning("No hay valores numéricos para esta métrica.")
        return

    desc = describe_series(s)
    norm = normality_test(s)

    st.markdown(f"#### Estadísticos descriptivos — *{metric_label}*")

    desc_df = pd.DataFrame(
        [
            {"Estadístico": k, "Valor": v}
            for k, v in desc.items()
        ]
    )
    desc_df["Valor"] = desc_df["Valor"].apply(
        lambda v: v if isinstance(v, (int, np.integer)) else (
            f"{v:.6f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else v
        )
    )
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.dataframe(desc_df, hide_index=True, width="stretch")
    with c_right:
        st.markdown("**Test de normalidad**")
        st.json(norm)
        if "p_value" in norm:
            verdict = (
                "✅ No se rechaza H₀ → distribución compatible con normal"
                if norm["p_value"] > 0.05
                else "❌ Se rechaza H₀ → distribución no normal"
            )
            st.caption(verdict + f" (α=0.05, {norm['test']})")

    # Plots: Histograma + KDE | Boxplot/Violin | Q-Q plot
    st.markdown("**Distribución**")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Histograma + KDE
    sns.histplot(s, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].axvline(desc["media"], color="#C44E52", linestyle="--", label=f"μ={desc['media']:.3f}")
    axes[0].axvline(desc["mediana"], color="#55A868", linestyle=":", label=f"med={desc['mediana']:.3f}")
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Histograma + KDE")
    axes[0].legend(fontsize=8)

    # 2) Boxplot (con violín superpuesto si hay agrupador)
    if by_var is not None and df_full is not None and by_var in df_full.columns:
        df_plot = df_full[[by_var]].copy()
        df_plot["__metric"] = pd.to_numeric(df_full[metric_values.name], errors="coerce")
        df_plot = df_plot.dropna()
        if not df_plot.empty:
            order = sorted(df_plot[by_var].dropna().unique().tolist(), key=str)
            sns.violinplot(
                data=df_plot,
                x=by_var,
                y="__metric",
                order=order,
                ax=axes[1],
                inner="box",
                cut=0,
            )
            axes[1].tick_params(axis="x", rotation=30)
            axes[1].set_ylabel(metric_label)
            axes[1].set_title(f"Distribución por {by_var}")
        else:
            sns.boxplot(y=s, ax=axes[1], color="#4C72B0")
            axes[1].set_title("Boxplot")
    else:
        sns.boxplot(y=s, ax=axes[1], color="#4C72B0")
        axes[1].set_ylabel(metric_label)
        axes[1].set_title("Boxplot")

    # 3) Q-Q plot
    sp_stats.probplot(s, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot vs Normal")
    axes[2].get_lines()[0].set_markerfacecolor("#4C72B0")
    axes[2].get_lines()[0].set_markeredgecolor("#4C72B0")
    axes[2].get_lines()[1].set_color("#C44E52")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_inference_categorical(
    df: pd.DataFrame, metric: str, metric_label: str, by: str, alpha: float
):
    res = compare_groups(df, metric, by, posthoc=True, posthoc_alpha=alpha)

    st.markdown(f"#### Inferencia — *{metric_label}* por **{by}**")

    if res.kind == "insufficient":
        st.warning(" / ".join(res.notes) or "Datos insuficientes.")
        if not res.descriptives.empty:
            st.dataframe(res.descriptives, hide_index=True, width="stretch")
        return

    st.caption(
        f"N total: **{res.n_total}** · niveles considerados: **{len(res.descriptives)}**"
    )

    st.markdown("**Descriptivos por grupo**")
    st.dataframe(
        res.descriptives,
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in res.descriptives.columns
            if res.descriptives[c].dtype.kind in "fc"
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test paramétrico**")
        p = res.parametric.get("p_value", float("nan"))
        st.json(res.parametric)
        st.caption(_significance_badge(p, alpha))
    with c2:
        st.markdown("**Test no paramétrico**")
        p = res.non_parametric.get("p_value", float("nan"))
        st.json(res.non_parametric)
        st.caption(_significance_badge(p, alpha))

    st.markdown("**Tamaño del efecto**")
    eff_df = pd.DataFrame(
        [
            {"Métrica": k, "Valor": (f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else v)}
            for k, v in res.effect_size.items()
        ]
    )
    st.dataframe(eff_df, hide_index=True, width="stretch")
    with st.expander("Cómo interpretar el tamaño del efecto", expanded=False):
        st.markdown(
            """
- **Cohen's d**: 0.2 pequeño · 0.5 mediano · 0.8 grande.
- **rank-biserial r**: rango [-1, 1]. |r|≈0.1 pequeño · 0.3 mediano · 0.5 grande.
- **η² / ε²**: 0.01 pequeño · 0.06 mediano · 0.14 grande (proporción de varianza).
            """
        )

    if res.posthoc is not None and not res.posthoc.empty:
        st.markdown("**Post-hoc — Mann-Whitney pareados (corrección de Bonferroni)**")
        st.dataframe(
            res.posthoc,
            hide_index=True,
            width="stretch",
            column_config={
                "U": st.column_config.NumberColumn(format="%.2f"),
                "p_value": st.column_config.NumberColumn(format="%.4f"),
                "p_Bonferroni": st.column_config.NumberColumn(format="%.4f"),
                "rank-biserial r": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    if res.notes:
        with st.expander("Notas", expanded=False):
            for n in res.notes:
                st.write(f"- {n}")

    st.download_button(
        "⬇ Descargar descriptivos (CSV)",
        data=res.descriptives.to_csv(index=False).encode("utf-8"),
        file_name=f"inferencia_{metric}_por_{by}_descriptivos.csv",
        mime="text/csv",
    )


def _render_inference_numeric(
    df: pd.DataFrame, metric: str, metric_label: str, var: str
):
    res, clean = correlation(df, metric, var)
    st.markdown(f"#### Correlación — *{metric_label}* vs **{var}**")
    if "error" in res:
        st.warning(res["error"])
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Pearson r",
            f"{res['Pearson r']:.4f}",
            help=f"p = {_format_pvalue(res['Pearson p_value'])}",
        )
        st.caption(_significance_badge(res["Pearson p_value"]))
    with c2:
        st.metric(
            "Spearman ρ",
            f"{res['Spearman ρ']:.4f}",
            help=f"p = {_format_pvalue(res['Spearman p_value'])}",
        )
        st.caption(_significance_badge(res["Spearman p_value"]))

    with st.expander("Regresión lineal simple", expanded=False):
        st.json(
            {
                "pendiente": res["regresión_pendiente"],
                "intercepto": res["regresión_intercepto"],
                "error estándar": res["regresión_se"],
                "p_value": res["regresión_p_value"],
                "R²": res["R²"],
            }
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=clean,
        x=var,
        y=metric,
        ax=ax,
        scatter_kws={"alpha": 0.6, "color": "#4C72B0"},
        line_kws={"color": "#C44E52"},
    )
    ax.set_xlabel(var)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {var}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------


def render_inference():
    s = ensure_state()
    ensure_groups_loaded_for_dataset(s.dataset_name)

    st.header("Inferencia y análisis estadístico detallado")
    st.caption(
        "Análisis descriptivo en profundidad sobre métricas concretas y "
        "tests inferenciales que relacionan dichas métricas con variables sociolinguísticas."
    )

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = _load_dataset(processed_dir, s.dataset_name)
    informantes_df = getattr(ds, "informantes", None)

    tema_names = sorted(ds.temas.keys())
    if not tema_names:
        st.warning("No hay temas disponibles en este dataset.")
        return

    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}
    group_names = list(st.session_state.groups.keys())

    # Selección base
    cA, cB, cC = st.columns([2, 2, 1])
    with cA:
        tema = st.selectbox("Tema", tema_names, key="inference::tema")
    with cB:
        active_group = st.selectbox(
            "Grupo de informantes (filtro previo)",
            group_names,
            key="inference::group",
        )
    with cC:
        alpha = st.number_input(
            "α (significación)",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.005,
            format="%.3f",
            key="inference::alpha",
        )

    # Filtrar tema por grupo
    group = st.session_state.groups.get(active_group, ALL_GROUP)
    informantes_f = (
        apply_group(informantes_df, group) if informantes_df is not None else None
    )
    df_tema_raw = ds.temas[tema]
    informant_col = _infer_informant_col(df_tema_raw)
    df_f = _filter_by_group(df_tema_raw, informantes_f, informant_col)

    cache_key = f"{s.dataset_name}::{tema}::{active_group}::{len(df_f)}"

    if df_f.empty:
        st.warning("No hay datos para este tema/grupo.")
        return

    # Construir tablas de métricas
    type_stats = _type_stats_cached(df_f, cache_key=cache_key + "::tok")
    inf_metrics_df = (
        _informant_metrics_cached(df_f, informantes_f, cache_key=cache_key + "::inf")
        if informantes_f is not None
        else _informant_metrics_cached(df_f, informantes_df, cache_key=cache_key + "::inf")
    )

    tabs = st.tabs(
        [
            "📊 Análisis descriptivo",
            "🧪 Inferencia (variables sociolinguísticas)",
            "🔗 Matriz de tests",
        ]
    )

    # =======================================================================
    # TAB 1 — Análisis descriptivo
    # =======================================================================
    with tabs[0]:
        st.subheader("Análisis descriptivo de una métrica concreta")

        c1, c2 = st.columns(2)
        with c1:
            scope = st.radio(
                "Nivel de agregación",
                ["Por informante", "Por type", "Por nodo del grafo"],
                key="inference::desc_scope",
                horizontal=True,
            )
        with c2:
            if scope == "Por type":
                metric_keys = list(TYPE_METRICS.keys())
                metric = st.selectbox(
                    "Métrica",
                    metric_keys,
                    format_func=lambda k: TYPE_METRICS[k],
                    key="inference::desc_metric_type",
                )
                source_df = type_stats
                metric_label = TYPE_METRICS[metric]
            elif scope == "Por nodo del grafo":
                directed = st.toggle(
                    "Grafo dirigido", value=True, key="inference::desc_directed"
                )
                metric_keys = list(NODE_METRICS.keys())
                metric = st.selectbox(
                    "Métrica",
                    metric_keys,
                    format_func=lambda k: NODE_METRICS[k],
                    key="inference::desc_metric_node",
                )
                source_df = _node_stats_cached(
                    df_f,
                    directed=directed,
                    cache_key=cache_key + f"::nod{directed}",
                )
                metric_label = NODE_METRICS[metric]
            else:  # Por informante
                metric_keys = list(INFORMANT_METRICS.keys())
                metric = st.selectbox(
                    "Métrica",
                    metric_keys,
                    format_func=lambda k: INFORMANT_METRICS[k],
                    key="inference::desc_metric_inf",
                )
                source_df = inf_metrics_df
                metric_label = INFORMANT_METRICS[metric]

        if metric not in source_df.columns:
            st.error(f"La métrica '{metric}' no está disponible.")
            return

        # Selector opcional de variable de agrupación (solo si es por informante)
        by_var = None
        if scope == "Por informante" and informantes_df is not None:
            cat_cols, _ = _categorize_variables(
                source_df,
                [c for c in source_df.columns if c not in INFORMANT_METRICS and c not in _INFORMANT_DROP_COLS],
            )
            if cat_cols:
                by_var = st.selectbox(
                    "Comparar distribución por (opcional)",
                    ["—"] + cat_cols,
                    key="inference::desc_by",
                )
                by_var = None if by_var == "—" else by_var

        _render_descriptive_section(
            source_df[metric],
            metric_label,
            by_var=by_var,
            df_full=source_df,
        )

        st.divider()
        st.markdown("**Datos de la métrica (primeras filas)**")
        st.dataframe(source_df.head(50), hide_index=True, width="stretch")

    # =======================================================================
    # TAB 2 — Inferencia
    # =======================================================================
    with tabs[1]:
        st.subheader("Tests inferenciales sobre variables sociolinguísticas")
        st.caption(
            "Las métricas se calculan **por informante** y se contrastan contra las "
            "variables disponibles en la tabla de informantes. Selecciona la "
            "variable categórica (comparación entre grupos) o numérica continua (correlación)."
        )

        if informantes_df is None:
            st.warning("Este dataset no contiene tabla de informantes.")
            return

        # Selección de métrica
        metric_keys = list(INFORMANT_METRICS.keys())
        metric = st.selectbox(
            "Métrica (por informante)",
            metric_keys,
            format_func=lambda k: INFORMANT_METRICS[k],
            key="inference::inf_metric",
        )
        metric_label = INFORMANT_METRICS[metric]

        # Variables disponibles (excluir las propias métricas)
        candidate_cols = [
            c
            for c in inf_metrics_df.columns
            if c not in INFORMANT_METRICS and c not in _INFORMANT_DROP_COLS
        ]
        cat_cols, num_cols = _categorize_variables(inf_metrics_df, candidate_cols)

        if not cat_cols and not num_cols:
            st.warning("No hay variables sociolinguísticas detectables.")
            return

        kind = st.radio(
            "Tipo de variable",
            (["Categórica (comparación de grupos)"] if cat_cols else [])
            + (["Numérica (correlación)"] if num_cols else []),
            key="inference::inf_kind",
            horizontal=True,
        )

        if kind.startswith("Categórica"):
            by = st.selectbox("Variable", cat_cols, key="inference::inf_by_cat")
            _render_inference_categorical(
                inf_metrics_df, metric, metric_label, by, alpha
            )
        else:
            var = st.selectbox("Variable", num_cols, key="inference::inf_by_num")
            _render_inference_numeric(inf_metrics_df, metric, metric_label, var)

    # =======================================================================
    # TAB 3 — Matriz de tests (vista global)
    # =======================================================================
    with tabs[2]:
        st.subheader("Matriz de p-values: métricas × variables")
        st.caption(
            "Vista global del nivel de significación de cada métrica frente a "
            "cada variable sociolinguística. Para variables categóricas se usa "
            "Kruskal-Wallis (o Mann-Whitney si solo hay 2 niveles). Para "
            "variables numéricas, correlación de Spearman."
        )

        if informantes_df is None:
            st.warning("Este dataset no contiene tabla de informantes.")
            return

        candidate_cols = [
            c
            for c in inf_metrics_df.columns
            if c not in INFORMANT_METRICS and c not in _INFORMANT_DROP_COLS
        ]
        cat_cols, num_cols = _categorize_variables(inf_metrics_df, candidate_cols)
        all_vars = cat_cols + num_cols
        if not all_vars:
            st.warning("No hay variables disponibles.")
            return

        sel_metrics: List[str] = st.multiselect(
            "Métricas",
            options=list(INFORMANT_METRICS.keys()),
            default=list(INFORMANT_METRICS.keys()),
            format_func=lambda k: INFORMANT_METRICS[k],
            key="inference::matrix_metrics",
        )
        sel_vars: List[str] = st.multiselect(
            "Variables",
            options=all_vars,
            default=all_vars,
            key="inference::matrix_vars",
        )

        if not sel_metrics or not sel_vars:
            st.info("Selecciona al menos una métrica y una variable.")
            return

        with st.spinner("Calculando matriz de tests..."):
            rows = []
            for m in sel_metrics:
                row = {"métrica": INFORMANT_METRICS[m]}
                for v in sel_vars:
                    if v in num_cols:
                        res, _ = correlation(inf_metrics_df, m, v)
                        p = res.get("Spearman p_value", np.nan)
                    else:
                        levels = inf_metrics_df[v].dropna().unique()
                        groups = [
                            pd.to_numeric(
                                inf_metrics_df.loc[inf_metrics_df[v] == lv, m],
                                errors="coerce",
                            )
                            .dropna()
                            .to_numpy()
                            for lv in levels
                        ]
                        groups = [g for g in groups if len(g) >= 2]
                        if len(groups) < 2:
                            p = np.nan
                        elif len(groups) == 2:
                            try:
                                _, p = sp_stats.mannwhitneyu(
                                    groups[0], groups[1], alternative="two-sided"
                                )
                            except Exception:
                                p = np.nan
                        else:
                            try:
                                _, p = sp_stats.kruskal(*groups)
                            except Exception:
                                p = np.nan
                    row[v] = p
                rows.append(row)
            matrix = pd.DataFrame(rows).set_index("métrica")

        # Render numérico + estilo (heatmap simple)
        st.markdown("**p-values**")

        def _color(v: float) -> str:
            if pd.isna(v):
                return "background-color: #eeeeee; color: #999;"
            if v < 0.001:
                return "background-color: #1a4f1a; color: white; font-weight: bold;"
            if v < 0.01:
                return "background-color: #4caf50; color: white; font-weight: bold;"
            if v < alpha:
                return "background-color: #a5d6a7;"
            return ""

        try:
            styled = matrix.style.format("{:.4f}").map(_color)
            st.dataframe(styled, width="stretch")
        except Exception:
            st.dataframe(matrix, width="stretch")

        st.caption(
            f"Resaltado: verde oscuro p<0.001, verde p<0.01, verde claro p<{alpha}, gris=N/D."
        )

        st.download_button(
            "⬇ Descargar matriz (CSV)",
            data=matrix.to_csv().encode("utf-8"),
            file_name=f"{s.dataset_name}_{tema}_{active_group}_matriz_pvalues.csv",
            mime="text/csv",
        )
