from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import networkx
import pandas as pd

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    graph_stats,
    node_stats,
)
from glaurlex.core.graph_service import GraphService
from glaurlex.core.groups import Group, apply_group
from glaurlex.core.groups_store import load_groups
from glaurlex.core.stats import estadisticas_df


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
    for col in candidates:
        if col in df_tema.columns:
            return col
    return None


def _slug_name(*parts: str) -> str:
    raw = "__".join(str(part) for part in parts if str(part))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "artifact"


def _filter_tema_by_group(
    df_tema: pd.DataFrame, informantes_df: pd.DataFrame, group: Group
) -> tuple[pd.DataFrame, str]:
    informantes_f = apply_group(informantes_df, group)
    informant_col = _infer_informant_col(df_tema)
    if informant_col is None:
        return df_tema, "sin_columna_informante"

    informant_id_col = "CODIGO_INFORMANTE" if "CODIGO_INFORMANTE" in informantes_f.columns else None
    if informant_id_col is None:
        allowed = set((informantes_f.index + 1).tolist())
    else:
        allowed = set(informantes_f[informant_id_col].tolist())
    return df_tema[df_tema[informant_col].isin(allowed)], "ok"


@dataclass
class GenerationSummary:
    stats_created: int = 0
    stats_skipped: int = 0
    graphs_created: int = 0
    graphs_skipped: int = 0
    graph_stats_created: int = 0
    graph_stats_skipped: int = 0


def _parse_csv(values: str | None) -> set[str] | None:
    if not values:
        return None
    out = {v.strip() for v in values.split(",") if v.strip()}
    return out or None


def generate_artifacts(
    dataset_name: str,
    processed_dir: Path,
    output_subdir: str,
    graph_mode: str,
    overwrite: bool,
    only_groups: set[str] | None,
    only_temas: set[str] | None,
) -> GenerationSummary:
    dataset_service = DatasetService(processed_dir)
    graph_service = GraphService(processed_dir)
    ds = dataset_service.load_processed(dataset_name)
    groups = load_groups(str(processed_dir), dataset_name)

    if only_groups is not None:
        groups = {name: group for name, group in groups.items() if name in only_groups}
        if not groups:
            raise ValueError("No quedan grupos tras aplicar --only-groups.")

    temas = sorted(ds.temas.keys())
    if only_temas is not None:
        temas = [tema for tema in temas if tema in only_temas]
        if not temas:
            raise ValueError("No quedan temas tras aplicar --only-temas.")

    output_dir = ds.root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    graph_stats_dir = output_dir / "graphs_stats"
    graph_stats_dir.mkdir(parents=True, exist_ok=True)

    summary = GenerationSummary()

    for group_name, group in groups.items():
        for tema in temas:
            df_tema = ds.temas[tema]
            df_tema_f, filter_status = _filter_tema_by_group(df_tema, ds.informantes, group)
            stats = estadisticas_df(df_tema_f)

            csv_name = _slug_name(dataset_name, tema, group_name, "estadisticas") + ".csv"
            csv_path = stats_dir / csv_name
            if csv_path.exists() and not overwrite:
                summary.stats_skipped += 1
            else:
                stats.to_csv(csv_path, index=False, encoding="utf-8")
                summary.stats_created += 1

            bigrams_df = bigrams_for_tema(df_tema_f)
            for directed in (True, False):
                if graph_mode == "dir" and not directed:
                    continue
                if graph_mode == "und" and directed:
                    continue

                graph_name = _slug_name(tema, group_name, "dir" if directed else "und")
                graph_path = graph_service.get_graph_path(dataset_name, graph_name)
                if graph_path.exists() and not overwrite:
                    summary.graphs_skipped += 1
                    graph = networkx.read_gml(graph_path)
                else:
                    graph = (
                        bigrams_to_dirgraph(bigrams_df)
                        if directed
                        else bigrams_to_undgraph(bigrams_df)
                    )
                    graph_service.save_graph(dataset_name, graph_name, graph, overwrite=True)
                    summary.graphs_created += 1

                # Estadísticas de la pestaña "Grafos" (sin métricas small-world)
                nstats = node_stats(graph)
                gstats = graph_stats(graph, nstats, include_small_world=False)

                mode = "dir" if directed else "und"
                node_stats_name = (
                    _slug_name(dataset_name, tema, group_name, mode, "node_stats") + ".csv"
                )
                graph_stats_name = (
                    _slug_name(dataset_name, tema, group_name, mode, "graph_stats") + ".csv"
                )

                node_stats_path = graph_stats_dir / node_stats_name
                graph_stats_path = graph_stats_dir / graph_stats_name

                if node_stats_path.exists() and not overwrite:
                    summary.graph_stats_skipped += 1
                else:
                    nstats.to_csv(node_stats_path, index=False, encoding="utf-8")
                    summary.graph_stats_created += 1

                if graph_stats_path.exists() and not overwrite:
                    summary.graph_stats_skipped += 1
                else:
                    pd.DataFrame([gstats]).to_csv(graph_stats_path, index=False, encoding="utf-8")
                    summary.graph_stats_created += 1

            print(
                f"[ok] group={group_name} tema={tema} rows={len(df_tema_f)} "
                f"stats={csv_name} filter={filter_status}"
            )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Genera automáticamente artefactos de visualize para todas las "
            "combinaciones grupo x tema de un dataset procesado."
        )
    )
    parser.add_argument("--dataset", required=True, help="Nombre del dataset procesado.")
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help=f"Directorio de datasets procesados. Default: {DEFAULT_PROCESSED_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        default="visualize_exports",
        help="Subdirectorio dentro del dataset para CSV de estadísticas.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=["both", "dir", "und"],
        default="both",
        help="Tipo de grafos a generar: ambos, dirigido o no dirigido.",
    )
    parser.add_argument(
        "--only-groups",
        default=None,
        help="Lista CSV de grupos a incluir. Ej: TODOS,B1",
    )
    parser.add_argument(
        "--only-temas",
        default=None,
        help="Lista CSV de temas a incluir. Ej: Animals,Town",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe CSV y GML existentes.",
    )
    args = parser.parse_args()

    summary = generate_artifacts(
        dataset_name=args.dataset,
        processed_dir=Path(args.processed_dir),
        output_subdir=args.output_subdir,
        graph_mode=args.graph_mode,
        overwrite=args.overwrite,
        only_groups=_parse_csv(args.only_groups),
        only_temas=_parse_csv(args.only_temas),
    )
    print(
        "[done] "
        f"stats_created={summary.stats_created} "
        f"stats_skipped={summary.stats_skipped} "
        f"graphs_created={summary.graphs_created} "
        f"graphs_skipped={summary.graphs_skipped} "
        f"graph_stats_created={summary.graph_stats_created} "
        f"graph_stats_skipped={summary.graph_stats_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
