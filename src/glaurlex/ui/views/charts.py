from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    community_leiden,
    node_stats,
)
from glaurlex.core.groups import ALL_GROUP, apply_group
from glaurlex.core.stats import estadisticas_df
from glaurlex.ui.state import (
    ensure_groups_loaded_for_dataset,
    ensure_state,
)

# ---------------------------------------------------------------------------
# Columnas disponibles para graficar
# ---------------------------------------------------------------------------

TOKEN_STATS_COLS: dict[str, str] = {
    "disponibilidad": "Disponibilidad",
    "aparición": "Aparición",
    "freq_rel": "Frecuencia relativa",
    "avg_pos": "Posición promedio",
    "freq_acum": "Frecuencia acumulada",
}

NODE_STATS_COLS: dict[str, str] = {
    "degree": "Grado",
    "strength": "Fuerza (grado ponderado)",
    "betweenness": "Intermediación",
    "closeness": "Cercanía",
    "pagerank": "PageRank",
    "eigenvector": "Eigenvector",
    "clustering": "Clustering",
    "degree_centrality": "Centralidad de grado",
}

ALL_METRIC_COLS: dict[str, str] = {**TOKEN_STATS_COLS, **NODE_STATS_COLS}

# Opciones para colorear puntos del scatter (variables categóricas)
COLOR_COLS: dict[str, str] = {
    "none": "Ninguno",
    "weak_component_id": "Componente conexa (débil)",
    "strong_component_id": "Componente conexa (fuerte, solo dirigido)",
    "community_id": "Comunidad (Leiden)",
}

SNS_STYLES = ["whitegrid", "darkgrid", "white", "dark", "ticks"]
SNS_PALETTES = [
    "deep",
    "muted",
    "pastel",
    "bright",
    "dark",
    "colorblind",
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Set3",
    "husl",
    "hls",
]

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
def _token_stats_cached(df_tema, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    return estadisticas_df(df_tema)


@st.cache_data(show_spinner=False)
def _node_stats_cached(df_tema, directed: bool, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    bigrams = bigrams_for_tema(df_tema)
    G = bigrams_to_dirgraph(bigrams) if directed else bigrams_to_undgraph(bigrams)
    return node_stats(G)


@st.cache_data(show_spinner=False)
def _community_cached(df_tema, directed: bool, cache_key: str) -> pd.DataFrame:
    """Detección de comunidades (Leiden) sobre el grafo no dirigido."""
    _ = cache_key
    bigrams = bigrams_for_tema(df_tema)
    G = bigrams_to_dirgraph(bigrams) if directed else bigrams_to_undgraph(bigrams)
    communities = community_leiden(G, seed=42)
    rows = [
        {"token": node, "community_id": i + 1}
        for i, community in enumerate(communities)
        for node in community
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_informant_col(df_tema: pd.DataFrame) -> str | None:
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
    df_tema: pd.DataFrame, informantes_f, informant_col: str | None
) -> pd.DataFrame:
    if informantes_f is None or informant_col is None:
        return df_tema
    id_col = "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
    allowed = set(
        (informantes_f.index + 1).tolist() if id_col is None else informantes_f[id_col].tolist()
    )
    return df_tema[df_tema[informant_col].isin(allowed)]


def _fig_to_bytes(fig: plt.Figure, fmt: str, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _download_row(fig: plt.Figure, prefix: str, dpi: int) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Descargar PNG",
            data=_fig_to_bytes(fig, "png", dpi),
            file_name=f"{prefix}.png",
            mime="image/png",
        )
    with c2:
        st.download_button(
            "⬇ Descargar SVG",
            data=_fig_to_bytes(fig, "svg", dpi),
            file_name=f"{prefix}.svg",
            mime="image/svg+xml",
        )


# ---------------------------------------------------------------------------
# Data loader (shared across chart types)
# ---------------------------------------------------------------------------


def _load_merged(
    ds,
    tema_name: str,
    group_name: str,
    source: str,
    directed: bool,
    dataset_name: str,
) -> pd.DataFrame:
    """Carga y fusiona estadísticas de tokens y/o nodos para un tema+grupo."""
    group = st.session_state.groups.get(group_name, ALL_GROUP)
    informantes_df = getattr(ds, "informantes", None)
    informantes_f = apply_group(informantes_df, group) if informantes_df is not None else None
    df_tema = ds.temas[tema_name]
    informant_col = _infer_informant_col(df_tema)
    df_f = _filter_by_group(df_tema, informantes_f, informant_col)
    cache_key = f"{dataset_name}::{tema_name}::{group_name}::{len(df_f)}"

    parts: list[pd.DataFrame] = []

    if source in ("Tokens", "Combinada"):
        tok = _token_stats_cached(df_f, cache_key=cache_key + "::tok")
        parts.append(tok.rename(columns={"token": "_id"}).set_index("_id"))

    if source in ("Nodos", "Combinada"):
        nod = _node_stats_cached(df_f, directed=directed, cache_key=cache_key + f"::nod{directed}")
        node_col = "node" if "node" in nod.columns else nod.columns[0]
        parts.append(nod.rename(columns={node_col: "_id"}).set_index("_id"))

    if len(parts) == 1:
        df = parts[0].reset_index().rename(columns={"_id": "token"})
    else:
        df = parts[0].join(parts[1], how="inner").reset_index().rename(columns={"_id": "token"})

    return df


def _load_multi_group(
    ds,
    tema_name: str,
    group_names: list[str],
    source: str,
    directed: bool,
    dataset_name: str,
) -> pd.DataFrame:
    """Carga datos para varios grupos y los concatena añadiendo columna 'Grupo'."""
    frames = []
    for g in group_names:
        df_g = _load_merged(ds, tema_name, g, source, directed, dataset_name)
        df_g = df_g.copy()
        df_g["Grupo"] = g
        frames.append(df_g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Color-by helper
# ---------------------------------------------------------------------------


def _attach_color_col(
    df_plot: pd.DataFrame,
    color_key: str,
    ds,
    tema: str,
    group_name: str,
    directed: bool,
    dataset_name: str,
) -> tuple[pd.DataFrame, str | None]:
    """Añade la columna de color al DataFrame y devuelve el nombre de la columna resultante.

    Retorna (df_plot enriquecido, nombre_columna) o (df_plot, None) si color_key == 'none'.
    """
    if color_key == "none":
        return df_plot, None

    # Recuperar el df filtrado del tema/grupo para poder calcular stats de nodos
    group = st.session_state.groups.get(group_name, ALL_GROUP)
    informantes_df = getattr(ds, "informantes", None)
    informantes_f = apply_group(informantes_df, group) if informantes_df is not None else None
    df_tema = ds.temas[tema]
    informant_col = _infer_informant_col(df_tema)
    df_f = _filter_by_group(df_tema, informantes_f, informant_col)
    cache_key = f"{dataset_name}::{tema}::{group_name}::{len(df_f)}"

    if color_key == "community_id":
        extra = _community_cached(df_f, directed, cache_key=cache_key + "::comm")
    else:
        # weak_component_id / strong_component_id vienen de node_stats
        nod = _node_stats_cached(df_f, directed, cache_key=cache_key + f"::nod{directed}")
        node_col = "node" if "node" in nod.columns else nod.columns[0]
        cols_to_keep = [node_col, color_key] if color_key in nod.columns else [node_col]
        extra = nod[cols_to_keep].rename(columns={node_col: "token"})

    if color_key not in extra.columns:
        return df_plot, None

    df_plot = df_plot.merge(extra[["token", color_key]], on="token", how="left")
    # Formato legible: entero → "C1", "C2", …
    df_plot[color_key] = (
        df_plot[color_key]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "?")
        .apply(lambda v: f"C{v}" if v != "?" else "?")
    )
    return df_plot, color_key


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_charts():
    s = ensure_state()
    ensure_groups_loaded_for_dataset(s.dataset_name)
    st.header("Gráficos")

    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    ds = _load_dataset(processed_dir, s.dataset_name)

    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}
    if "active_group" not in st.session_state:
        st.session_state.active_group = "TODOS"

    tema_names = sorted(ds.temas.keys())
    group_names = list(st.session_state.groups.keys())

    if not tema_names:
        st.warning("No hay temas disponibles en este dataset.")
        return

    # -----------------------------------------------------------------------
    # Configuración general
    # -----------------------------------------------------------------------
    st.subheader("Configuración")

    col_type, col_source = st.columns(2)
    with col_type:
        chart_type = st.radio(
            "Tipo de gráfico",
            ["Barras", "Scatter", "Histograma", "Boxplot"],
            horizontal=True,
            key="charts::chart_type",
        )
    with col_source:
        source = st.radio(
            "Fuente de métricas",
            ["Tokens", "Nodos", "Combinada"],
            horizontal=True,
            key="charts::source",
        )

    directed = False
    if source in ("Nodos", "Combinada"):
        directed = st.toggle(
            "Grafo dirigido (para métricas de nodos)", value=True, key="charts::directed"
        )

    available_cols = (
        TOKEN_STATS_COLS
        if source == "Tokens"
        else NODE_STATS_COLS
        if source == "Nodos"
        else ALL_METRIC_COLS
    )
    col_keys = list(available_cols.keys())

    st.divider()

    # -----------------------------------------------------------------------
    # Selección de datos y variables (varía por tipo de gráfico)
    # -----------------------------------------------------------------------

    if chart_type == "Barras":
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            tema = st.selectbox("Tema", tema_names, key="charts::bar_tema")
        with c2:
            bar_groups = st.multiselect(
                "Grupos",
                group_names,
                default=[group_names[0]],
                key="charts::bar_groups",
            )
        with c3:
            bar_all = st.checkbox("Todos", value=False, key="charts::bar_all")
            top_n = (
                None
                if bar_all
                else st.number_input("Top N tokens", 5, 200, 20, key="charts::bar_top_n")
            )

        multi_group = len(bar_groups) > 1
        if multi_group:
            st.caption(
                "Comparación multi-grupo: los tokens del eje X se ordenan según el **primer grupo** seleccionado."
            )
            y_cols = st.multiselect(
                "Métrica (eje Y)",
                options=col_keys,
                format_func=lambda k: available_cols[k],
                default=[col_keys[0]],
                max_selections=1,
                key="charts::bar_y",
            )
        else:
            y_cols = st.multiselect(
                "Métricas (eje Y) — selecciona una o varias",
                options=col_keys,
                format_func=lambda k: available_cols[k],
                default=[col_keys[0]],
                key="charts::bar_y",
            )

    elif chart_type == "Scatter":
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            tema = st.selectbox("Tema", tema_names, key="charts::scat_tema")
        with c2:
            scat_groups = st.multiselect(
                "Grupos",
                group_names,
                default=[group_names[0]],
                key="charts::scat_groups",
            )
        with c3:
            scat_all = st.checkbox("Todos", value=False, key="charts::scat_all")
            top_n = (
                None
                if scat_all
                else st.number_input("Top N tokens", 5, 500, 100, key="charts::scat_top_n")
            )

        cx, cy = st.columns(2)
        with cx:
            x_col = st.selectbox(
                "Variable X",
                col_keys,
                format_func=lambda k: available_cols[k],
                index=0,
                key="charts::scat_x",
            )
        with cy:
            y_col = st.selectbox(
                "Variable Y",
                col_keys,
                format_func=lambda k: available_cols[k],
                index=min(1, len(col_keys) - 1),
                key="charts::scat_y",
            )
        c_labels, c_trend, c_color = st.columns(3)
        with c_labels:
            show_labels = st.toggle(
                "Mostrar etiquetas de tokens", value=False, key="charts::scat_labels"
            )
        with c_trend:
            show_trend = st.toggle(
                "Línea de tendencia (regresión lineal)", value=False, key="charts::scat_trend"
            )
        with c_color:
            multi_scat = len(st.session_state.get("charts::scat_groups", [group_names[0]])) > 1
            color_key = st.selectbox(
                "Color por",
                options=list(COLOR_COLS.keys()),
                format_func=lambda k: COLOR_COLS[k],
                key="charts::scat_color",
                disabled=multi_scat,
                help="No disponible en modo multi-grupo (el color ya representa el grupo)."
                if multi_scat
                else None,
            )

    elif chart_type == "Histograma":
        c1, c2 = st.columns(2)
        with c1:
            tema = st.selectbox("Tema", tema_names, key="charts::hist_tema")
        with c2:
            hist_groups = st.multiselect(
                "Grupos",
                group_names,
                default=[group_names[0]],
                key="charts::hist_groups",
            )

        cx, cy = st.columns(2)
        with cx:
            hist_col = st.selectbox(
                "Variable",
                col_keys,
                format_func=lambda k: available_cols[k],
                key="charts::hist_col",
            )
        with cy:
            bins = st.slider("Número de bins", 5, 100, 30, key="charts::hist_bins")

    else:  # Boxplot
        compare_by = st.radio(
            "Comparar por",
            ["Temas", "Grupos", "Temas × Grupos"],
            horizontal=True,
            key="charts::box_compare_by",
        )

        if compare_by == "Temas":
            c1, c2 = st.columns([3, 1])
            with c1:
                box_temas = st.multiselect(
                    "Temas a comparar",
                    tema_names,
                    default=tema_names[: min(3, len(tema_names))],
                    key="charts::box_temas",
                )
            with c2:
                box_single_group = st.selectbox("Grupo", group_names, key="charts::box_group")
        elif compare_by == "Grupos":
            c1, c2 = st.columns([1, 3])
            with c1:
                box_single_tema = st.selectbox("Tema", tema_names, key="charts::box_tema")
            with c2:
                box_groups = st.multiselect(
                    "Grupos a comparar",
                    group_names,
                    default=group_names[: min(3, len(group_names))],
                    key="charts::box_groups",
                )
        else:  # Temas × Grupos
            c1, c2 = st.columns(2)
            with c1:
                box_temas = st.multiselect(
                    "Temas (eje X)",
                    tema_names,
                    default=tema_names[: min(3, len(tema_names))],
                    key="charts::box_temas",
                )
            with c2:
                box_groups = st.multiselect(
                    "Grupos (color)",
                    group_names,
                    default=group_names[: min(3, len(group_names))],
                    key="charts::box_groups",
                )

        box_col = st.selectbox(
            "Variable a comparar",
            col_keys,
            format_func=lambda k: available_cols[k],
            key="charts::box_col",
        )

    # -----------------------------------------------------------------------
    # Opciones de exportación
    # -----------------------------------------------------------------------
    with st.expander("Opciones de estilo y exportación", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            sns_style = st.selectbox("Estilo", SNS_STYLES, index=0, key="charts::style")
            primary_color = st.color_picker(
                "Color principal",
                value="#4C72B0",
                key="charts::primary_color",
                help="Color para gráficos de una sola serie (barras con una métrica, etc.)",
            )
        with ec2:
            sns_palette = st.selectbox(
                "Paleta de colores", SNS_PALETTES, index=0, key="charts::palette"
            )
            palette_colors = sns.color_palette(sns_palette, n_colors=10).as_hex()
            swatches = "".join(
                f'<span style="display:inline-block;width:20px;height:20px;'
                f'background:{c};border-radius:3px;margin-right:2px;"></span>'
                for c in palette_colors
            )
            st.markdown(swatches, unsafe_allow_html=True)
        with ec3:
            fig_w = st.slider("Ancho (pulgadas)", 4, 20, 10, key="charts::fig_w")
            fig_h = st.slider("Alto (pulgadas)", 3, 15, 5, key="charts::fig_h")
        with ec4:
            dpi = st.slider("DPI (PNG)", 72, 600, 300, step=50, key="charts::dpi")

    # -----------------------------------------------------------------------
    # Generar gráfico
    # -----------------------------------------------------------------------
    st.divider()
    if not st.button("Generar gráfico", type="primary", key="charts::run"):
        return

    sns.set_theme(style=sns_style)
    sns.set_palette(sns_palette)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    prefix = f"{s.dataset_name}_{chart_type.lower()}"

    try:
        with st.spinner("Calculando datos..."):
            # --- Barras ---
            if chart_type == "Barras":
                if not y_cols:
                    st.warning("Selecciona al menos una métrica.")
                    plt.close(fig)
                    return
                if not bar_groups:
                    st.warning("Selecciona al menos un grupo.")
                    plt.close(fig)
                    return

                multi_group = len(bar_groups) > 1
                metric = y_cols[0]

                if multi_group:
                    # Referencia de tokens: top-N del primer grupo
                    df_ref = _load_merged(ds, tema, bar_groups[0], source, directed, s.dataset_name)
                    if metric not in df_ref.columns:
                        st.error(f"La columna '{metric}' no está disponible.")
                        plt.close(fig)
                        return
                    ref_tokens = (
                        df_ref.sort_values(metric, ascending=False)["token"].tolist()
                        if top_n is None
                        else df_ref.nlargest(int(top_n), metric)["token"].tolist()
                    )

                    frames = []
                    for g in bar_groups:
                        df_g = _load_merged(ds, tema, g, source, directed, s.dataset_name)
                        df_g = df_g[df_g["token"].isin(ref_tokens)][["token", metric]].copy()
                        df_g["Grupo"] = g
                        frames.append(df_g)
                    df_plot = pd.concat(frames, ignore_index=True)
                    # Mantener orden del ranking de referencia
                    df_plot["token"] = pd.Categorical(
                        df_plot["token"], categories=ref_tokens, ordered=True
                    )
                    df_plot = df_plot.sort_values("token")

                    sns.barplot(data=df_plot, x="token", y=metric, hue="Grupo", ax=ax)
                    ax.set_ylabel(available_cols.get(metric, metric))
                    top_label = "Todos" if top_n is None else f"Top {top_n}"
                    ax.set_title(
                        f"{top_label} — {tema} | "
                        + ", ".join(bar_groups)
                        + f"\n(orden: {bar_groups[0]})"
                    )
                    prefix += f"_{tema}_multigrupo"

                else:
                    # Un solo grupo: permite multi-métrica
                    df = _load_merged(ds, tema, bar_groups[0], source, directed, s.dataset_name)
                    if metric not in df.columns:
                        st.error(f"La columna '{metric}' no está disponible.")
                        plt.close(fig)
                        return
                    df_plot = df if top_n is None else df.nlargest(int(top_n), metric)

                    if len(y_cols) == 1:
                        sns.barplot(data=df_plot, x="token", y=metric, ax=ax, color=primary_color)
                        ax.set_ylabel(available_cols[metric])
                    else:
                        valid_y = [c for c in y_cols if c in df_plot.columns]
                        df_melt = df_plot[["token"] + valid_y].melt(
                            id_vars=["token"], var_name="métrica", value_name="valor"
                        )
                        df_melt["métrica"] = df_melt["métrica"].map(
                            lambda k: available_cols.get(k, k)
                        )
                        sns.barplot(data=df_melt, x="token", y="valor", hue="métrica", ax=ax)
                        ax.set_ylabel("Valor")

                    top_label = "Todos" if top_n is None else f"Top {top_n}"
                    ax.set_title(f"{top_label} — {tema} / {bar_groups[0]}")
                    prefix += f"_{tema}_{bar_groups[0]}"

                ax.set_xlabel("Token")
                ax.tick_params(axis="x", rotation=45)

            # --- Scatter ---
            elif chart_type == "Scatter":
                if not scat_groups:
                    st.warning("Selecciona al menos un grupo.")
                    plt.close(fig)
                    return

                if len(scat_groups) == 1:
                    df = _load_merged(ds, tema, scat_groups[0], source, directed, s.dataset_name)
                    if x_col not in df.columns or y_col not in df.columns:
                        st.error("Una o ambas columnas seleccionadas no están disponibles.")
                        plt.close(fig)
                        return
                    df_plot = df.dropna(subset=[x_col, y_col])
                    if top_n is not None:
                        df_plot = df_plot.head(int(top_n))

                    df_plot, hue_col = _attach_color_col(
                        df_plot, color_key, ds, tema, scat_groups[0], directed, s.dataset_name
                    )

                    if show_trend and hue_col is None:
                        # regplot solo disponible sin hue categórico
                        sns.regplot(
                            data=df_plot,
                            x=x_col,
                            y=y_col,
                            ax=ax,
                            scatter_kws={"alpha": 0.7},
                            line_kws={"linewidth": 1.5},
                        )
                    elif show_trend and hue_col is not None:
                        # scatter coloreado + líneas de tendencia por categoría
                        categories = df_plot[hue_col].unique()
                        palette = sns.color_palette(n_colors=len(categories))
                        sns.scatterplot(
                            data=df_plot,
                            x=x_col,
                            y=y_col,
                            hue=hue_col,
                            palette=palette,
                            ax=ax,
                            alpha=0.7,
                        )
                        for color, cat in zip(palette, categories):
                            df_cat = df_plot[df_plot[hue_col] == cat]
                            if len(df_cat) >= 2:
                                sns.regplot(
                                    data=df_cat,
                                    x=x_col,
                                    y=y_col,
                                    ax=ax,
                                    scatter=False,
                                    color=color,
                                    line_kws={"linewidth": 1.2, "linestyle": "--"},
                                )
                    else:
                        sns.scatterplot(
                            data=df_plot,
                            x=x_col,
                            y=y_col,
                            hue=hue_col,
                            ax=ax,
                            alpha=0.7,
                        )

                    if show_labels:
                        for _, row in df_plot.iterrows():
                            ax.annotate(
                                str(row["token"]),
                                (row[x_col], row[y_col]),
                                fontsize=7,
                                alpha=0.75,
                                ha="center",
                                va="bottom",
                            )

                    hue_label = COLOR_COLS.get(color_key, "") if hue_col else ""
                    ax.set_title(
                        f"{available_cols.get(x_col, x_col)} vs "
                        f"{available_cols.get(y_col, y_col)} — {tema} / {scat_groups[0]}"
                        + (f"\nColor: {hue_label}" if hue_col else "")
                    )
                    if hue_col:
                        ax.legend(title=hue_label, bbox_to_anchor=(1.01, 1), loc="upper left")
                else:
                    combined = _load_multi_group(
                        ds, tema, scat_groups, source, directed, s.dataset_name
                    )
                    if x_col not in combined.columns or y_col not in combined.columns:
                        st.error("Una o ambas columnas seleccionadas no están disponibles.")
                        plt.close(fig)
                        return
                    df_plot = combined.dropna(subset=[x_col, y_col])
                    if top_n is not None:
                        # Top N por grupo (para no saturar el gráfico)
                        df_plot = pd.concat(
                            [
                                grp.nlargest(int(top_n), x_col)
                                for _, grp in df_plot.groupby("Grupo")
                            ],
                            ignore_index=True,
                        )
                    palette = sns.color_palette(n_colors=len(scat_groups))
                    sns.scatterplot(
                        data=df_plot,
                        x=x_col,
                        y=y_col,
                        hue="Grupo",
                        palette=palette,
                        ax=ax,
                        alpha=0.7,
                    )
                    if show_trend:
                        for color, group_name in zip(palette, scat_groups):
                            df_g = df_plot[df_plot["Grupo"] == group_name]
                            sns.regplot(
                                data=df_g,
                                x=x_col,
                                y=y_col,
                                ax=ax,
                                scatter=False,
                                color=color,
                                line_kws={"linewidth": 1.5, "linestyle": "--"},
                            )
                    if show_labels:
                        for _, row in df_plot.iterrows():
                            ax.annotate(
                                str(row["token"]),
                                (row[x_col], row[y_col]),
                                fontsize=7,
                                alpha=0.6,
                                ha="center",
                                va="bottom",
                            )
                    ax.set_title(
                        f"{available_cols.get(x_col, x_col)} vs "
                        f"{available_cols.get(y_col, y_col)} — {tema} | " + ", ".join(scat_groups)
                    )

                ax.set_xlabel(available_cols.get(x_col, x_col))
                ax.set_ylabel(available_cols.get(y_col, y_col))
                prefix += f"_{x_col}_vs_{y_col}"

            # --- Histograma ---
            elif chart_type == "Histograma":
                if not hist_groups:
                    st.warning("Selecciona al menos un grupo.")
                    plt.close(fig)
                    return

                if len(hist_groups) == 1:
                    df = _load_merged(ds, tema, hist_groups[0], source, directed, s.dataset_name)
                    if hist_col not in df.columns:
                        st.error(f"La columna '{hist_col}' no está disponible.")
                        plt.close(fig)
                        return
                    sns.histplot(data=df, x=hist_col, bins=bins, ax=ax, kde=True)
                    ax.set_title(
                        f"Distribución de {available_cols.get(hist_col, hist_col)} "
                        f"— {tema} / {hist_groups[0]}"
                    )
                else:
                    combined = _load_multi_group(
                        ds, tema, hist_groups, source, directed, s.dataset_name
                    )
                    if hist_col not in combined.columns:
                        st.error(f"La columna '{hist_col}' no está disponible.")
                        plt.close(fig)
                        return
                    sns.histplot(
                        data=combined,
                        x=hist_col,
                        hue="Grupo",
                        bins=bins,
                        ax=ax,
                        kde=True,
                        element="step",
                        fill=False,
                    )
                    ax.set_title(
                        f"Distribución de {available_cols.get(hist_col, hist_col)} "
                        f"— {tema} | " + ", ".join(hist_groups)
                    )

                ax.set_xlabel(available_cols.get(hist_col, hist_col))
                prefix += f"_{hist_col}"

            # --- Boxplot ---
            elif chart_type == "Boxplot":
                if compare_by == "Temas":
                    if not box_temas:
                        st.warning("Selecciona al menos un tema.")
                        plt.close(fig)
                        return
                    frames = []
                    for t in box_temas:
                        df_t = _load_merged(
                            ds, t, box_single_group, source, directed, s.dataset_name
                        )
                        if box_col in df_t.columns:
                            frames.append(df_t[[box_col]].assign(Tema=t))
                    if not frames:
                        st.error("Ningún tema tiene la columna seleccionada.")
                        plt.close(fig)
                        return
                    combined = pd.concat(frames, ignore_index=True)
                    sns.boxplot(data=combined, x="Tema", y=box_col, ax=ax)
                    ax.tick_params(axis="x", rotation=30)
                    ax.set_ylabel(available_cols.get(box_col, box_col))
                    ax.set_title(
                        f"Boxplot de {available_cols.get(box_col, box_col)} por tema"
                        f" — {box_single_group}"
                    )
                    prefix += f"_boxplot_{box_col}_temas"

                elif compare_by == "Grupos":
                    if not box_groups:
                        st.warning("Selecciona al menos un grupo.")
                        plt.close(fig)
                        return
                    frames = []
                    for g in box_groups:
                        df_g = _load_merged(
                            ds, box_single_tema, g, source, directed, s.dataset_name
                        )
                        if box_col in df_g.columns:
                            frames.append(df_g[[box_col]].assign(Grupo=g))
                    if not frames:
                        st.error("Ningún grupo tiene la columna seleccionada.")
                        plt.close(fig)
                        return
                    combined = pd.concat(frames, ignore_index=True)
                    sns.boxplot(data=combined, x="Grupo", y=box_col, ax=ax)
                    ax.tick_params(axis="x", rotation=30)
                    ax.set_ylabel(available_cols.get(box_col, box_col))
                    ax.set_title(
                        f"Boxplot de {available_cols.get(box_col, box_col)} por grupo"
                        f" — {box_single_tema}"
                    )
                    prefix += f"_boxplot_{box_col}_grupos"

                else:  # Temas × Grupos
                    if not box_temas or not box_groups:
                        st.warning("Selecciona al menos un tema y un grupo.")
                        plt.close(fig)
                        return
                    frames = []
                    for t in box_temas:
                        for g in box_groups:
                            df_tg = _load_merged(ds, t, g, source, directed, s.dataset_name)
                            if box_col in df_tg.columns:
                                frames.append(df_tg[[box_col]].assign(Tema=t, Grupo=g))
                    if not frames:
                        st.error("Ninguna combinación tema/grupo tiene la columna seleccionada.")
                        plt.close(fig)
                        return
                    combined = pd.concat(frames, ignore_index=True)
                    sns.boxplot(data=combined, x="Tema", y=box_col, hue="Grupo", ax=ax)
                    ax.tick_params(axis="x", rotation=30)
                    ax.set_ylabel(available_cols.get(box_col, box_col))
                    ax.set_title(
                        f"Boxplot de {available_cols.get(box_col, box_col)} — temas × grupos"
                    )
                    prefix += f"_boxplot_{box_col}_temas_x_grupos"

        fig.tight_layout()
        st.pyplot(fig)
        _download_row(fig, prefix, dpi)

    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        raise
    finally:
        plt.close(fig)
