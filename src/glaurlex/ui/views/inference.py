"""! @package glaurlex.ui.views.inference
Vista de análisis estadístico descriptivo e inferencial sobre métricas
léxicas y variables sociolinguísticas.
"""

from __future__ import annotations

import io
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats as sp_stats

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.core.inference import (
    compare_groups,
    correlation,
    describe_series,
    normality_test,
)
from glaurlex.core.metrics_catalog import labels_by_scope
from glaurlex.core.variables_store import get_order, is_ordinal
from glaurlex.ui.metrics_cache import (
    filter_by_group,
    infer_informant_col,
    informant_metrics_cached,
    node_stats_cached,
    type_stats_cached,
)
from glaurlex.ui.state import (
    ensure_groups_loaded_for_dataset,
    ensure_state,
    ensure_variables_loaded_for_dataset,
)

# ---------------------------------------------------------------------------
# Catálogos de métricas (derivados del catálogo central en core)
# ---------------------------------------------------------------------------

INFORMANT_METRICS: dict[str, str] = labels_by_scope("informant")
TYPE_METRICS: dict[str, str] = labels_by_scope("type")
NODE_METRICS: dict[str, str] = labels_by_scope("node")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _fig_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    """! Serializa una figura matplotlib a PNG en alta resolución."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


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
    order: Optional[List[str]] = None,
):
    s = pd.to_numeric(metric_values, errors="coerce").dropna()
    if s.empty:
        st.warning("No hay valores numéricos para esta métrica.")
        return

    desc = describe_series(s)
    norm = normality_test(s)

    st.markdown(f"#### Estadísticos descriptivos — *{metric_label}*")

    desc_df = pd.DataFrame([{"Estadístico": k, "Valor": v} for k, v in desc.items()])
    desc_df["Valor"] = desc_df["Valor"].apply(
        lambda v: v
        if isinstance(v, (int, np.integer))
        else (f"{v:.6f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else v)
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
    # Cada gráfico se define como un "drawer" que pinta sobre un Axes dado, para
    # poder renderizarlo tanto en la vista combinada como en figuras individuales
    # de alta resolución descargables.
    def _draw_hist(ax: plt.Axes) -> None:
        sns.histplot(s, kde=True, ax=ax, color="#4C72B0")
        ax.axvline(desc["media"], color="#C44E52", linestyle="--", label=f"μ={desc['media']:.3f}")
        ax.axvline(
            desc["mediana"], color="#55A868", linestyle=":", label=f"med={desc['mediana']:.3f}"
        )
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Frecuencia")
        ax.set_title("Histograma + KDE")
        ax.legend(fontsize=8)

    def _draw_box(ax: plt.Axes) -> None:
        if by_var is not None and df_full is not None and by_var in df_full.columns:
            df_plot = df_full[[by_var]].copy()
            df_plot["__metric"] = pd.to_numeric(df_full[metric_values.name], errors="coerce")
            df_plot = df_plot.dropna()
            if not df_plot.empty:
                present = df_plot[by_var].dropna().astype(str).unique().tolist()
                if order:
                    seen = set()
                    plot_order = [
                        lv for lv in order if lv in present and not (lv in seen or seen.add(lv))
                    ]
                    plot_order += sorted(lv for lv in present if lv not in set(order))
                else:
                    plot_order = sorted(present, key=str)
                df_plot[by_var] = df_plot[by_var].astype(str)
                sns.violinplot(
                    data=df_plot,
                    x=by_var,
                    y="__metric",
                    order=plot_order,
                    ax=ax,
                    inner="box",
                    cut=0,
                )
                ax.tick_params(axis="x", rotation=30)
                ax.set_ylabel(metric_label)
                ax.set_title(f"Distribución por {by_var}")
                return
        sns.boxplot(y=s, ax=ax, color="#4C72B0")
        ax.set_ylabel(metric_label)
        ax.set_title("Boxplot")

    def _draw_qq(ax: plt.Axes) -> None:
        sp_stats.probplot(s, dist="norm", plot=ax)
        ax.set_title("Q-Q plot vs Normal")
        ax.get_lines()[0].set_markerfacecolor("#4C72B0")
        ax.get_lines()[0].set_markeredgecolor("#4C72B0")
        ax.get_lines()[1].set_color("#C44E52")

    st.markdown("**Distribución**")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _draw_hist(axes[0])
    _draw_box(axes[1])
    _draw_qq(axes[2])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Descarga individual de cada gráfico en alta resolución (PNG, 300 dpi).
    metric_key = str(metric_values.name) if metric_values.name is not None else "metrica"
    plots = [
        ("histograma", "Histograma + KDE", _draw_hist),
        ("distribucion", "Boxplot/Violín", _draw_box),
        ("qqplot", "Q-Q plot", _draw_qq),
    ]
    dl_cols = st.columns(3)
    for (tag, btn_label, drawer), col in zip(plots, dl_cols):
        single_fig, single_ax = plt.subplots(figsize=(7, 5))
        drawer(single_ax)
        single_fig.tight_layout()
        with col:
            st.download_button(
                f"⬇ {btn_label} (PNG)",
                data=_fig_to_png_bytes(single_fig, dpi=300),
                file_name=f"{metric_key}_{tag}.png",
                mime="image/png",
                key=f"inference::desc_dl_{tag}",
            )
        plt.close(single_fig)


def _render_inference_categorical(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    by: str,
    alpha: float,
    order: Optional[List[str]] = None,
):
    res = compare_groups(df, metric, by, posthoc=True, posthoc_alpha=alpha, order=order)

    st.markdown(f"#### Inferencia — *{metric_label}* por **{by}**")
    if order:
        st.caption("Variable marcada como **ordinal** — orden: " + " < ".join(order))

    if res.kind == "insufficient":
        st.warning(" / ".join(res.notes) or "Datos insuficientes.")
        if not res.descriptives.empty:
            st.dataframe(res.descriptives, hide_index=True, width="stretch")
        return

    st.caption(f"N total: **{res.n_total}** · niveles considerados: **{len(res.descriptives)}**")

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
            {
                "Métrica": k,
                "Valor": (
                    f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else v
                ),
            }
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

    if res.trend:
        st.markdown("**Tendencia (variable ordinal)**")
        if "error" in res.trend:
            st.caption(res.trend["error"])
        else:
            t = res.trend
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Spearman ρ",
                    f"{t['Spearman ρ']:.4f}",
                    help=f"p = {_format_pvalue(t['Spearman p_value'])}",
                )
                st.caption(_significance_badge(t["Spearman p_value"], alpha))
            with c2:
                st.metric(
                    "Kendall τ-b",
                    f"{t['Kendall τ-b']:.4f}",
                    help=f"p = {_format_pvalue(t['Kendall p_value'])}",
                )
                st.caption(_significance_badge(t["Kendall p_value"], alpha))
            st.caption(
                "Test de tendencia monotónica sobre rank(nivel) vs métrica. "
                "Significativo ⇒ la métrica crece (o decrece) sistemáticamente "
                "con el orden declarado."
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


def _render_inference_numeric(df: pd.DataFrame, metric: str, metric_label: str, var: str):
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
    ensure_variables_loaded_for_dataset(s.dataset_name)
    variables_cfg: dict = st.session_state.get("variables", {}) or {}

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
    informantes_f = apply_group(informantes_df, group) if informantes_df is not None else None
    df_tema_raw = ds.temas[tema]
    informant_col = infer_informant_col(df_tema_raw)
    df_f = filter_by_group(df_tema_raw, informantes_f, informant_col)

    cache_key = f"{s.dataset_name}::{tema}::{active_group}::{len(df_f)}"

    if df_f.empty:
        st.warning("No hay datos para este tema/grupo.")
        return

    # Construir tablas de métricas (los wrappers cacheados se reutilizan
    # entre vistas, así type_stats no se recomputa al construir las
    # métricas por informante).
    type_stats = type_stats_cached(df_f, cache_key=cache_key)
    inf_metrics_df = informant_metrics_cached(
        df_f,
        informantes_f if informantes_f is not None else informantes_df,
        cache_key=cache_key,
    )

    TAB_LABELS = [
        "Análisis descriptivo",
        "Inferencia sobre variables sociolinguísticas",
        "Matriz sobre variables sociolinguísticas",
        "Inferencia sobre grupos",
        "Matriz sobre grupos",
    ]
    if st.session_state.get("inference::active_tab") not in TAB_LABELS:
        st.session_state["inference::active_tab"] = TAB_LABELS[0]

    def _keep_inference_tab_selected():
        val = st.session_state.get("inference::active_tab")
        if val in TAB_LABELS:
            st.session_state["inference::_active_tab_prev"] = val
        else:
            st.session_state["inference::active_tab"] = st.session_state.get(
                "inference::_active_tab_prev", TAB_LABELS[0]
            )

    active_tab = st.segmented_control(
        "Vista",
        options=TAB_LABELS,
        key="inference::active_tab",
        label_visibility="collapsed",
        on_change=_keep_inference_tab_selected,
    )
    if active_tab not in TAB_LABELS:
        active_tab = st.session_state.get("inference::_active_tab_prev", TAB_LABELS[0])

    # =======================================================================
    # TAB 1 — Análisis descriptivo
    # =======================================================================
    if active_tab == TAB_LABELS[0]:
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
                directed = st.toggle("Grafo dirigido", value=True, key="inference::desc_directed")
                metric_keys = list(NODE_METRICS.keys())
                metric = st.selectbox(
                    "Métrica",
                    metric_keys,
                    format_func=lambda k: NODE_METRICS[k],
                    key="inference::desc_metric_node",
                )
                source_df = node_stats_cached(
                    df_f,
                    directed=directed,
                    cache_key=cache_key,
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
                [
                    c
                    for c in source_df.columns
                    if c not in INFORMANT_METRICS and c not in _INFORMANT_DROP_COLS
                ],
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
            order=get_order(variables_cfg, by_var) if by_var else None,
        )

        st.divider()
        st.markdown("**Datos de la métrica (primeras filas)**")
        st.dataframe(source_df.head(50), hide_index=True, width="stretch")
        scope_tag = scope.replace(" ", "_").lower()
        st.download_button(
            "⬇ Descargar datos de la métrica (CSV)",
            data=source_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{s.dataset_name}_{tema}_{active_group}_{scope_tag}_{metric}_datos.csv",
            mime="text/csv",
            key="inference::desc_data_dl",
        )

    # =======================================================================
    # TAB 2 — Inferencia
    # =======================================================================
    elif active_tab == TAB_LABELS[1]:
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
                inf_metrics_df,
                metric,
                metric_label,
                by,
                alpha,
                order=get_order(variables_cfg, by) or None,
            )
        else:
            var = st.selectbox("Variable", num_cols, key="inference::inf_by_num")
            _render_inference_numeric(inf_metrics_df, metric, metric_label, var)

    # =======================================================================
    # TAB 3 — Matriz de tests (vista global)
    # =======================================================================
    elif active_tab == TAB_LABELS[2]:
        st.subheader("Matriz de p-values: métricas × variables")
        st.caption(
            "Vista global del nivel de significación de cada métrica frente a "
            "cada variable sociolinguística. Para variables categóricas se usa "
            "Kruskal-Wallis (o Mann-Whitney si solo hay 2 niveles). Para "
            "variables numéricas, correlación de Spearman. **Variables "
            "ordinales** (configuradas en la pestaña *Variables*) se evalúan "
            "como Spearman sobre rank(nivel)."
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
                    v_is_ordinal = is_ordinal(variables_cfg, v)
                    if v in num_cols:
                        res, _ = correlation(inf_metrics_df, m, v)
                        p = res.get("Spearman p_value", np.nan)
                    elif v_is_ordinal:
                        # Codificar nivel -> rank y correr Spearman.
                        order = get_order(variables_cfg, v)
                        rank_idx = {str(lv): i for i, lv in enumerate(order)}
                        ranks = inf_metrics_df[v].astype(str).map(rank_idx)
                        encoded = inf_metrics_df.assign(__ord=ranks)
                        res, _ = correlation(encoded, m, "__ord")
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

    # =======================================================================
    # TAB 4 — Comparativa de stats de type entre grupos (filtros)
    # =======================================================================
    elif active_tab == TAB_LABELS[3]:
        st.subheader("Comparación de métricas por type entre grupos")
        st.caption(
            "Selecciona dos o más grupos (filtros guardados sobre informantes) "
            "y una métrica por type. Para cada grupo se recalculan las "
            "estadísticas por type a partir del tema actual y se contrastan "
            "las distribuciones entre grupos."
        )

        if informantes_df is None:
            st.warning("Este dataset no contiene tabla de informantes.")
            return

        if len(group_names) < 2:
            st.info(
                "Necesitas al menos dos grupos guardados (incluyendo TODOS) "
                "para comparar. Crea grupos en la pestaña *Grupos*."
            )
            return

        default_sel = [g for g in group_names if g != "TODOS"][:3]
        if len(default_sel) < 2:
            default_sel = group_names[:2]

        sel_groups: List[str] = st.multiselect(
            "Grupos a comparar",
            options=group_names,
            default=default_sel,
            key="inference::cmp_groups",
        )

        SOURCE_LABELS = {
            "stats": "Estadísticas por type",
            "graph_und": "Métricas de grafo (no dirigido)",
            "graph_dir": "Métricas de grafo (dirigido)",
        }
        cS, c1, c2 = st.columns([2, 2, 1])
        with cS:
            cmp_source = st.radio(
                "Fuente de métricas",
                list(SOURCE_LABELS.keys()),
                format_func=lambda k: SOURCE_LABELS[k],
                key="inference::cmp_source",
                horizontal=False,
            )
        with c1:
            if cmp_source == "stats":
                metric_catalog = TYPE_METRICS
                cmp_metric = st.selectbox(
                    "Métrica por type",
                    list(TYPE_METRICS.keys()),
                    format_func=lambda k: TYPE_METRICS[k],
                    key="inference::cmp_metric_type",
                )
            else:
                metric_catalog = NODE_METRICS
                cmp_metric = st.selectbox(
                    "Métrica por nodo",
                    list(NODE_METRICS.keys()),
                    format_func=lambda k: NODE_METRICS[k],
                    key="inference::cmp_metric_node",
                )
        with c2:
            min_groups_per_type = st.number_input(
                "Mín. grupos donde aparece el type",
                min_value=2,
                max_value=max(2, len(sel_groups) if sel_groups else 2),
                value=2,
                step=1,
                key="inference::cmp_min_groups",
                help=(
                    "Para la tabla pivote y la comparación pareada se "
                    "consideran solo los types presentes en al menos este "
                    "número de grupos seleccionados."
                ),
            )

        if len(sel_groups) < 2:
            st.info("Selecciona al menos dos grupos.")
            return

        cmp_metric_label = metric_catalog[cmp_metric]
        directed = cmp_source == "graph_dir"
        is_graph = cmp_source != "stats"

        # Construir tabla por grupo (type_stats o node_stats según fuente)
        per_group_stats: dict[str, pd.DataFrame] = {}
        per_group_n: dict[str, int] = {}
        spinner_msg = (
            f"Calculando métricas de grafo ({'dirigido' if directed else 'no dirigido'}) por grupo..."
            if is_graph
            else "Calculando estadísticas por grupo..."
        )
        with st.spinner(spinner_msg):
            for gname in sel_groups:
                g = st.session_state.groups.get(gname, ALL_GROUP)
                inf_g = apply_group(informantes_df, g)
                df_g = filter_by_group(df_tema_raw, inf_g, informant_col)
                gkey = f"{s.dataset_name}::{tema}::{gname}::{len(df_g)}"
                if is_graph:
                    ns_g = node_stats_cached(df_g, directed=directed, cache_key=gkey)
                    if not ns_g.empty and "node" in ns_g.columns:
                        ns_g = ns_g.rename(columns={"node": "type"})
                    per_group_stats[gname] = ns_g
                else:
                    per_group_stats[gname] = type_stats_cached(df_g, cache_key=gkey)
                per_group_n[gname] = (
                    int(df_g[informant_col].nunique()) if informant_col and not df_g.empty else 0
                )

        # Resumen N por grupo
        st.markdown("**Tamaño efectivo por grupo**")
        types_col_label = "nodos del grafo" if is_graph else "types únicos"
        size_rows = []
        for gname in sel_groups:
            ts_g = per_group_stats[gname]
            size_rows.append(
                {
                    "grupo": gname,
                    "informantes": per_group_n.get(gname, 0),
                    types_col_label: int(ts_g.shape[0]),
                }
            )
        st.dataframe(pd.DataFrame(size_rows), hide_index=True, width="stretch")

        # Long-form: type, __group, métrica
        long_rows = []
        for gname, ts_g in per_group_stats.items():
            if cmp_metric not in ts_g.columns or ts_g.empty:
                continue
            sub = ts_g[["type", cmp_metric]].copy()
            sub["__group"] = gname
            long_rows.append(sub)
        if not long_rows:
            st.warning("Ninguno de los grupos seleccionados tiene datos para esta métrica.")
            return

        long_df = pd.concat(long_rows, ignore_index=True)
        long_df[cmp_metric] = pd.to_numeric(long_df[cmp_metric], errors="coerce")
        long_df = long_df.dropna(subset=[cmp_metric])

        if long_df.empty or long_df["__group"].nunique() < 2:
            st.warning("Datos insuficientes para comparar entre grupos.")
            return

        # Inferencia: comparar la distribución de la métrica entre grupos
        cmp_res = compare_groups(
            long_df,
            cmp_metric,
            "__group",
            posthoc=True,
            posthoc_alpha=alpha,
            order=sel_groups,
        )

        st.markdown(f"#### Distribución de *{cmp_metric_label}* por grupo")

        # Boxplot/violin
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(sel_groups) + 4), 5))
        present_groups = [g for g in sel_groups if g in long_df["__group"].unique()]
        sns.violinplot(
            data=long_df,
            x="__group",
            y=cmp_metric,
            order=present_groups,
            ax=ax,
            inner="box",
            cut=0,
        )
        ax.set_xlabel("Grupo")
        ax.set_ylabel(cmp_metric_label)
        ax.set_title(f"{cmp_metric_label} por grupo — tema «{tema}»")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if cmp_res.kind == "insufficient":
            st.warning(" / ".join(cmp_res.notes) or "Datos insuficientes.")
        else:
            st.markdown("**Descriptivos por grupo (sobre la métrica por type)**")
            st.dataframe(
                cmp_res.descriptives.rename(columns={"__group": "grupo"}),
                hide_index=True,
                width="stretch",
                column_config={
                    c: st.column_config.NumberColumn(format="%.4f")
                    for c in cmp_res.descriptives.columns
                    if cmp_res.descriptives[c].dtype.kind in "fc"
                },
            )

            cP, cN = st.columns(2)
            with cP:
                st.markdown("**Test paramétrico**")
                p = cmp_res.parametric.get("p_value", float("nan"))
                st.json(cmp_res.parametric)
                st.caption(_significance_badge(p, alpha))
            with cN:
                st.markdown("**Test no paramétrico**")
                p = cmp_res.non_parametric.get("p_value", float("nan"))
                st.json(cmp_res.non_parametric)
                st.caption(_significance_badge(p, alpha))

            st.markdown("**Tamaño del efecto**")
            eff_df = pd.DataFrame(
                [
                    {
                        "Métrica": k,
                        "Valor": (
                            f"{v:.4f}"
                            if isinstance(v, (int, float, np.floating)) and not pd.isna(v)
                            else v
                        ),
                    }
                    for k, v in cmp_res.effect_size.items()
                ]
            )
            st.dataframe(eff_df, hide_index=True, width="stretch")

            if cmp_res.posthoc is not None and not cmp_res.posthoc.empty:
                st.markdown("**Post-hoc — Mann-Whitney pareados (corrección de Bonferroni)**")
                ph = cmp_res.posthoc.rename(columns={"grupo_1": "grupo_A", "grupo_2": "grupo_B"})
                st.dataframe(
                    ph,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "U": st.column_config.NumberColumn(format="%.2f"),
                        "p_value": st.column_config.NumberColumn(format="%.4f"),
                        "p_Bonferroni": st.column_config.NumberColumn(format="%.4f"),
                        "rank-biserial r": st.column_config.NumberColumn(format="%.3f"),
                    },
                )

            if cmp_res.notes:
                with st.expander("Notas", expanded=False):
                    for n in cmp_res.notes:
                        st.write(f"- {n}")

        st.divider()

        # Tabla pivote por type x grupo
        st.markdown(f"#### Comparativa por type — *{cmp_metric_label}*")
        pivot = long_df.pivot_table(
            index="type",
            columns="__group",
            values=cmp_metric,
            aggfunc="first",
        ).reindex(columns=present_groups)
        n_present = pivot.notna().sum(axis=1)
        pivot = pivot.loc[n_present >= int(min_groups_per_type)]

        if pivot.empty:
            st.info(
                "No hay types que cumplan el mínimo de grupos seleccionado. "
                "Reduce el umbral para ver más filas."
            )
        else:
            # Métrica de variación entre grupos (max - min) para ordenar
            pivot = pivot.assign(
                __max=pivot.max(axis=1, skipna=True),
                __range=pivot.max(axis=1, skipna=True) - pivot.min(axis=1, skipna=True),
            )

            sort_mode = st.radio(
                "Ordenar por",
                ["Mayor diferencia entre grupos", "Mayor valor", "Alfabético"],
                key="inference::cmp_sort",
                horizontal=True,
            )
            top_n = st.slider(
                "Top N types",
                min_value=10,
                max_value=min(500, max(10, int(pivot.shape[0]))),
                value=min(50, int(pivot.shape[0])),
                step=10,
                key="inference::cmp_topn",
            )

            if sort_mode.startswith("Mayor diferencia"):
                pivot = pivot.sort_values("__range", ascending=False)
            elif sort_mode.startswith("Mayor valor"):
                pivot = pivot.sort_values("__max", ascending=False)
            else:
                pivot = pivot.sort_index()

            display = pivot.drop(columns=["__max", "__range"]).head(top_n)
            st.dataframe(
                display,
                width="stretch",
                column_config={
                    c: st.column_config.NumberColumn(format="%.4f") for c in display.columns
                },
            )

            source_tag = (
                "grafo_dir"
                if cmp_source == "graph_dir"
                else "grafo_und"
                if cmp_source == "graph_und"
                else "type"
            )
            st.download_button(
                "⬇ Descargar comparativa por type (CSV)",
                data=pivot.drop(columns=["__max", "__range"]).to_csv().encode("utf-8"),
                file_name=(
                    f"{s.dataset_name}_{tema}_{source_tag}_{cmp_metric}_comparativa_grupos.csv"
                ),
                mime="text/csv",
            )

    # =======================================================================
    # TAB 5 — Matriz de p-values: métricas × comparaciones entre grupos
    # =======================================================================
    elif active_tab == TAB_LABELS[4]:
        st.subheader("Matriz de p-values: métricas × comparaciones entre grupos")
        st.caption(
            "Vista global del nivel de significación al comparar cada métrica "
            "entre los grupos seleccionados. La columna **Global** usa "
            "Kruskal-Wallis (≥3 grupos) o Mann-Whitney (2 grupos). Las "
            "columnas siguientes son comparaciones pareadas con Mann-Whitney."
        )

        if informantes_df is None:
            st.warning("Este dataset no contiene tabla de informantes.")
            return

        if len(group_names) < 2:
            st.info(
                "Necesitas al menos dos grupos guardados (incluyendo TODOS) "
                "para comparar. Crea grupos en la pestaña *Grupos*."
            )
            return

        MG_SOURCE_LABELS = {
            "informant": "Métricas por informante",
            "stats": "Estadísticas por type",
            "graph_und": "Métricas de grafo (no dirigido)",
            "graph_dir": "Métricas de grafo (dirigido)",
        }
        MG_CATALOGS = {
            "informant": INFORMANT_METRICS,
            "stats": TYPE_METRICS,
            "graph_und": NODE_METRICS,
            "graph_dir": NODE_METRICS,
        }

        cMS, cMG = st.columns([2, 3])
        with cMS:
            mg_source = st.radio(
                "Fuente de métricas",
                list(MG_SOURCE_LABELS.keys()),
                format_func=lambda k: MG_SOURCE_LABELS[k],
                key="inference::matrix_groups_source",
            )
        with cMG:
            default_mg_groups = [g for g in group_names if g != "TODOS"][:3]
            if len(default_mg_groups) < 2:
                default_mg_groups = group_names[:2]
            mg_sel_groups: List[str] = st.multiselect(
                "Grupos a comparar",
                options=group_names,
                default=default_mg_groups,
                key="inference::matrix_groups_groups",
            )

        mg_catalog = MG_CATALOGS[mg_source]
        mg_sel_metrics: List[str] = st.multiselect(
            "Métricas",
            options=list(mg_catalog.keys()),
            default=list(mg_catalog.keys()),
            format_func=lambda k: mg_catalog[k],
            key="inference::matrix_groups_metrics",
        )

        mg_show_pairwise = st.checkbox(
            "Incluir comparaciones pareadas",
            value=True,
            key="inference::matrix_groups_pairwise",
            help=(
                "Añade una columna por cada par de grupos con el p-value de "
                "Mann-Whitney (sin corrección de múltiples comparaciones)."
            ),
        )

        if len(mg_sel_groups) < 2:
            st.info("Selecciona al menos dos grupos.")
            return
        if not mg_sel_metrics:
            st.info("Selecciona al menos una métrica.")
            return

        mg_directed = mg_source == "graph_dir"
        mg_is_graph = mg_source.startswith("graph")
        mg_is_informant = mg_source == "informant"

        per_group_mdf: dict[str, pd.DataFrame] = {}
        per_group_n_mg: dict[str, int] = {}
        spinner_msg_mg = (
            f"Calculando métricas de grafo ({'dirigido' if mg_directed else 'no dirigido'}) por grupo..."
            if mg_is_graph
            else (
                "Calculando métricas por informante por grupo..."
                if mg_is_informant
                else "Calculando estadísticas por type por grupo..."
            )
        )
        with st.spinner(spinner_msg_mg):
            for gname in mg_sel_groups:
                g = st.session_state.groups.get(gname, ALL_GROUP)
                inf_g = apply_group(informantes_df, g)
                df_g = filter_by_group(df_tema_raw, inf_g, informant_col)
                gkey = f"{s.dataset_name}::{tema}::{gname}::{len(df_g)}"
                if mg_is_graph:
                    per_group_mdf[gname] = node_stats_cached(
                        df_g, directed=mg_directed, cache_key=gkey
                    )
                elif mg_is_informant:
                    per_group_mdf[gname] = informant_metrics_cached(
                        df_g,
                        inf_g if inf_g is not None else informantes_df,
                        cache_key=gkey,
                    )
                else:
                    per_group_mdf[gname] = type_stats_cached(df_g, cache_key=gkey)
                per_group_n_mg[gname] = (
                    int(df_g[informant_col].nunique()) if informant_col and not df_g.empty else 0
                )

        # Resumen N por grupo
        st.markdown("**Tamaño efectivo por grupo**")
        n_label = (
            "informantes (filas)"
            if mg_is_informant
            else "nodos del grafo"
            if mg_is_graph
            else "types únicos"
        )
        size_rows_mg = []
        for gname in mg_sel_groups:
            gdf = per_group_mdf.get(gname, pd.DataFrame())
            size_rows_mg.append(
                {
                    "grupo": gname,
                    "informantes": per_group_n_mg.get(gname, 0),
                    n_label: int(gdf.shape[0]),
                }
            )
        st.dataframe(pd.DataFrame(size_rows_mg), hide_index=True, width="stretch")

        with st.spinner("Calculando matriz de tests..."):
            rows_mg = []
            for m in mg_sel_metrics:
                row = {"métrica": mg_catalog[m]}

                groups_vals: dict[str, np.ndarray] = {}
                for gname in mg_sel_groups:
                    gdf = per_group_mdf.get(gname, pd.DataFrame())
                    if gdf.empty or m not in gdf.columns:
                        continue
                    vals = pd.to_numeric(gdf[m], errors="coerce").dropna().to_numpy()
                    if len(vals) >= 2:
                        groups_vals[gname] = vals

                valid = list(groups_vals.values())
                if len(valid) < 2:
                    row["Global"] = np.nan
                elif len(valid) == 2:
                    try:
                        _, p = sp_stats.mannwhitneyu(valid[0], valid[1], alternative="two-sided")
                        row["Global"] = float(p)
                    except Exception:
                        row["Global"] = np.nan
                else:
                    try:
                        _, p = sp_stats.kruskal(*valid)
                        row["Global"] = float(p)
                    except Exception:
                        row["Global"] = np.nan

                if mg_show_pairwise:
                    present = [g for g in mg_sel_groups if g in groups_vals]
                    for i in range(len(present)):
                        for j in range(i + 1, len(present)):
                            ga, gb = present[i], present[j]
                            try:
                                _, p_pair = sp_stats.mannwhitneyu(
                                    groups_vals[ga],
                                    groups_vals[gb],
                                    alternative="two-sided",
                                )
                                row[f"{ga} vs {gb}"] = float(p_pair)
                            except Exception:
                                row[f"{ga} vs {gb}"] = np.nan

                rows_mg.append(row)
            matrix_mg = pd.DataFrame(rows_mg).set_index("métrica")

        st.markdown("**p-values**")

        def _color_mg(v: float) -> str:
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
            styled_mg = matrix_mg.style.format("{:.4f}").map(_color_mg)
            st.dataframe(styled_mg, width="stretch")
        except Exception:
            st.dataframe(matrix_mg, width="stretch")

        st.caption(
            f"Resaltado: verde oscuro p<0.001, verde p<0.01, verde claro p<{alpha}, gris=N/D. "
            "Las comparaciones pareadas no incluyen corrección por múltiples tests."
        )

        mg_source_tag = {
            "informant": "informante",
            "stats": "type",
            "graph_und": "grafo_und",
            "graph_dir": "grafo_dir",
        }[mg_source]
        st.download_button(
            "⬇ Descargar matriz por grupos (CSV)",
            data=matrix_mg.to_csv().encode("utf-8"),
            file_name=(f"{s.dataset_name}_{tema}_{mg_source_tag}_matriz_pvalues_grupos.csv"),
            mime="text/csv",
        )
