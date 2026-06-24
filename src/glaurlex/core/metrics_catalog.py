"""! @package glaurlex.core.metrics_catalog
Catálogo central de métricas léxicas y de grafo del proyecto.

Proporciona una única fuente de verdad para los metadatos de cada métrica
(clave estable, nombre legible, descripción, algoritmo, parámetros,
ubicación de la implementación y prerrequisitos de cómputo). La UI consume
este catálogo para construir selectores y etiquetas, y sirve de base para
generar documentación automática de métricas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricScope = Literal["informant", "type", "node"]

# Prerrequisitos de cómputo. Modelados como literales para facilitar
# validación y documentación; ampliables sin romper consumidores existentes.
MetricRequirement = Literal[
    "tokens",  # responses/tokens dataframe (mínimo común)
    "type_stats",  # necesita estadisticas_df() previa (p.ej. disponibilidad)
    "graph",  # necesita un grafo construido (dirigido o no)
    "graph_directed",  # requiere específicamente un grafo dirigido
]


@dataclass(frozen=True)
class MetricSpec:
    """! Especificación de una métrica del catálogo.

    Attributes:
        - `key`: clave estable usada como identificador (n_types, ttr, ...).
        - `name`: nombre legible para UI.
        - `description`: descripción breve para tooltips/docs.
        - `scope`: ámbito de la métrica (informant, type, node).
        - `algorithm`: descripción textual del algoritmo (1-3 frases).
        - `parameters`: parámetros relevantes del cálculo (texto libre).
        - `implementation`: referencia tipo "module.function" donde se computa.
        - `dependencies`: prerrequisitos de cómputo. Tupla vacía si solo se
          necesitan los tokens crudos (caso por defecto).
    """

    key: str
    name: str
    description: str
    scope: MetricScope
    algorithm: str
    parameters: str = ""
    implementation: str = ""
    dependencies: tuple[MetricRequirement, ...] = ("tokens",)


def _spec(
    key: str,
    name: str,
    description: str,
    scope: MetricScope,
    algorithm: str,
    *,
    parameters: str = "",
    implementation: str = "",
    dependencies: tuple[MetricRequirement, ...] = ("tokens",),
) -> tuple[str, MetricSpec]:
    return key, MetricSpec(
        key=key,
        name=name,
        description=description,
        scope=scope,
        algorithm=algorithm,
        parameters=parameters,
        implementation=implementation,
        dependencies=dependencies,
    )


# ---------------------------------------------------------------------------
# Registro central
# ---------------------------------------------------------------------------
# Orden: por scope (informant → type → node) y, dentro de cada scope, en el
# mismo orden que aparecen actualmente en la UI para preservar la experiencia.

METRIC_CATALOG: dict[str, MetricSpec] = dict(
    [
        # --- Informant ---
        _spec(
            "n_tokens",
            "Tokens producidos",
            "Número total de respuestas (tokens) producidas por el informante.",
            "informant",
            "Cuenta de filas asociadas al informante en el dataframe del tema.",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "n_types",
            "Types distintos",
            "Número de types distintos producidos por el informante.",
            "informant",
            "Conteo de valores únicos de la columna `type` por informante.",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "type_coverage",
            "Cobertura de types",
            "Proporción de types distintos del tema cubiertos por el informante.",
            "informant",
            "n_types del informante / nº total de types distintos del tema (0 si el tema no tiene types).",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "ttr",
            "Type-Token Ratio (TTR)",
            "Ratio de diversidad léxica: types distintos / tokens producidos.",
            "informant",
            "n_types / n_tokens (0 si no hay tokens).",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "mean_pos",
            "Posición media",
            "Posición media (1-indexed) de las respuestas del informante.",
            "informant",
            "Media de la columna `pos` por informante, sumando 1 para indexar desde 1.",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "max_pos",
            "Longitud (max_pos)",
            "Última posición alcanzada por el informante (longitud de la lista).",
            "informant",
            "Máximo de `pos` por informante + 1.",
            implementation="glaurlex.core.inference.informant_metrics",
        ),
        _spec(
            "entropy",
            "Entropía de Shannon",
            "Entropía de Shannon (bits) sobre la distribución de types del informante.",
            "informant",
            "H = -Σ p_i · log2(p_i) sobre las frecuencias de cada type.",
            parameters="Base logarítmica: 2 (bits).",
            implementation="glaurlex.core.inference._shannon_entropy",
        ),
        _spec(
            "total_disp",
            "Disponibilidad acumulada",
            "Suma de la disponibilidad de los types producidos por el informante.",
            "informant",
            "Suma de los valores de `disponibilidad` (precalculados sobre el grupo activo) para cada token producido.",
            parameters="La disponibilidad se calcula sobre el grupo activo, no globalmente.",
            implementation="glaurlex.core.inference.informant_metrics",
            dependencies=("tokens", "type_stats"),
        ),
        _spec(
            "mean_disp",
            "Disponibilidad media",
            "Media de la disponibilidad de los types producidos por el informante.",
            "informant",
            "Promedio de los valores de `disponibilidad` para cada token producido.",
            parameters="La disponibilidad se calcula sobre el grupo activo.",
            implementation="glaurlex.core.inference.informant_metrics",
            dependencies=("tokens", "type_stats"),
        ),
        # --- Type ---
        _spec(
            "disponibilidad",
            "Disponibilidad",
            "Disponibilidad léxica del type (relevancia ponderada por posición y alcance).",
            "type",
            "Σ_pos exp(-2.3 · pos / pos_max) · (informantes_que_lo_dijeron_en_pos / total_informantes).",
            parameters="Decaimiento exponencial con factor -2.3 normalizado por la posición máxima del tema.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        _spec(
            "aparición",
            "Aparición",
            "Proporción de informantes que mencionaron el type al menos una vez.",
            "type",
            "nº de informantes únicos que produjeron el type / nº total de informantes.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        _spec(
            "freq_rel",
            "Frecuencia relativa",
            "Proporción de tokens del type sobre el total de tokens del tema.",
            "type",
            "value_counts(normalize=True) sobre la columna `type`.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        _spec(
            "avg_pos",
            "Posición promedio",
            "Posición promedio (0-indexed) en la que el type es mencionado.",
            "type",
            "Media de `pos` agrupada por type.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        _spec(
            "freq_acum",
            "Frecuencia acumulada",
            "Frecuencia relativa acumulada al ordenar los types por disponibilidad descendente.",
            "type",
            "Suma corrida de `freq_rel` tras ordenar por `disponibilidad` desc.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        _spec(
            "tokens",
            "Tokens",
            "Número total de ocurrencias del type en el dataset.",
            "type",
            "value_counts() sobre la columna `type`.",
            implementation="glaurlex.core.stats.estadisticas_df",
        ),
        # --- Node (graph) ---
        _spec(
            "degree",
            "Grado",
            "Número de vecinos (aristas incidentes) del nodo en el grafo.",
            "node",
            "nx.Graph.degree() sin pesos.",
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "degree_centrality",
            "Centralidad de grado",
            "Centralidad de grado normalizada por (N-1).",
            "node",
            "nx.degree_centrality(graph).",
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "strength",
            "Fuerza (grado ponderado)",
            "Suma de los pesos de las aristas incidentes al nodo.",
            "node",
            'nx.Graph.degree(weight="weight").',
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "betweenness",
            "Intermediación",
            "Fracción de caminos mínimos que atraviesan el nodo (sin normalizar).",
            "node",
            "nx.betweenness_centrality(graph, normalized=False).",
            parameters="normalized=False",
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "closeness",
            "Cercanía",
            "Inversa de la distancia media a los demás nodos (mejora de Wasserman-Faust).",
            "node",
            "nx.closeness_centrality(graph, wf_improved=True).",
            parameters="wf_improved=True",
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "pagerank",
            "PageRank",
            "Importancia del nodo según el algoritmo PageRank ponderado.",
            "node",
            'nx.pagerank(graph, weight="weight", alpha=0.85, tol=1e-6).',
            parameters='alpha=0.85, tol=1e-6, weight="weight"',
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "eigenvector",
            "Eigenvector",
            "Centralidad eigenvector ponderada (autovector dominante de la matriz de adyacencia).",
            "node",
            'nx.eigenvector_centrality(graph, weight="weight", max_iter=1000).',
            parameters='max_iter=1000, weight="weight"',
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
        _spec(
            "clustering",
            "Clustering",
            "Coeficiente de agrupamiento ponderado del nodo.",
            "node",
            'nx.clustering(graph, weight="weight").',
            parameters='weight="weight"',
            implementation="glaurlex.core.graph.node_stats",
            dependencies=("graph",),
        ),
    ]
)


# ---------------------------------------------------------------------------
# Helpers públicos
# ---------------------------------------------------------------------------


def get_metric(key: str) -> MetricSpec:
    """! Devuelve la `MetricSpec` asociada a una clave.

    @raises KeyError si la clave no está registrada.
    """
    return METRIC_CATALOG[key]


def metrics_by_scope(scope: MetricScope) -> dict[str, MetricSpec]:
    """! Devuelve el subconjunto de métricas para un scope dado, preservando el orden."""
    return {k: m for k, m in METRIC_CATALOG.items() if m.scope == scope}


def labels_by_scope(scope: MetricScope) -> dict[str, str]:
    """! Devuelve `{key: name}` para el scope — sustituye los dicts duplicados de la UI."""
    return {k: m.name for k, m in METRIC_CATALOG.items() if m.scope == scope}


def metrics_requiring(req: MetricRequirement) -> dict[str, MetricSpec]:
    """! Métricas que dependen de un prerrequisito dado (p.ej. `"graph"`).

    Útil para que la UI decida si debe construir un grafo o calcular las stats
    por type antes de mostrar selectores, o para avisar al usuario.
    """
    return {k: m for k, m in METRIC_CATALOG.items() if req in m.dependencies}
