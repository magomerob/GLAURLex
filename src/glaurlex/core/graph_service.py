"""! @package glaurlex.core.graph_service
Servicios para almacenar y cargar grafos en formato GML.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx


class GraphService:
    """! Servicio de grafos.

    - guarda grafos en GML dentro del directorio del dataset
    - lista grafos guardados
    - carga grafos desde GML
    """

    def __init__(self, processed_dir: str | Path) -> None:
        """! Crea el servicio y asegura el directorio base.

        @param processed_dir Directorio base donde se guardan datasets procesados.
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_graphs_dir(self, dataset_name: str) -> Path:
        """! Devuelve el directorio `graphs` de un dataset."""
        return self.processed_dir / dataset_name / "graphs"

    def list_graphs(self, dataset_name: str) -> List[str]:
        """! Lista grafos disponibles (archivos .gml).

        @param dataset_name Nombre del dataset.
        @return Lista de nombres de grafos (sin extensión).
        """
        graphs_dir = self.get_graphs_dir(dataset_name)
        if not graphs_dir.exists():
            return []
        return [f.stem for f in sorted(graphs_dir.glob("*.gml"))]

    def get_graph_path(self, dataset_name: str, graph_name: str) -> Path:
        """! Construye la ruta al archivo .gml de un grafo."""
        self._validate_graph_name(graph_name)
        return self.get_graphs_dir(dataset_name) / f"{graph_name}.gml"

    def save_graph(
        self,
        dataset_name: str,
        graph_name: str,
        graph: nx.Graph,
        overwrite: bool = False,
    ) -> Path:
        """! Guarda un grafo en formato GML.

        @param dataset_name Nombre del dataset.
        @param graph_name Nombre lógico del grafo (sin extensión).
        @param graph Grafo NetworkX a guardar.
        @param overwrite Si True, sobrescribe el archivo existente.
        @return Ruta del archivo .gml escrito.
        @exception FileNotFoundError Si el dataset no existe.
        @exception FileExistsError Si el grafo ya existe y overwrite es False.
        """
        self._validate_dataset_dir(dataset_name)
        self._validate_graph_name(graph_name)

        graphs_dir = self.get_graphs_dir(dataset_name)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        path = graphs_dir / f"{graph_name}.gml"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Ya existe el grafo '{graph_name}' en {graphs_dir}. "
                "Usa overwrite=True si quieres sobrescribirlo."
            )

        nx.write_gml(graph, path)
        return path

    def load_graph(self, dataset_name: str, graph_name: str) -> nx.Graph:
        """! Carga un grafo desde un archivo GML.

        @param dataset_name Nombre del dataset.
        @param graph_name Nombre lógico del grafo (sin extensión).
        @return Grafo NetworkX cargado.
        @exception FileNotFoundError Si el archivo no existe.
        """
        path = self.get_graph_path(dataset_name, graph_name)
        if not path.exists():
            raise FileNotFoundError(f"No existe el grafo '{graph_name}' en {path.parent}")
        return nx.read_gml(path)

    def _validate_dataset_dir(self, dataset_name: str) -> None:
        root = self.processed_dir / dataset_name
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    @staticmethod
    def _validate_graph_name(graph_name: str) -> None:
        if Path(graph_name).name != graph_name:
            raise ValueError("graph_name no puede contener separadores de ruta.")
