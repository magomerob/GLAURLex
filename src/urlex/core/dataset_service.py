from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from urlex.core.models import LoadResult


@dataclass(frozen=True)
class ProcessedDataset:
    """
    Representa un dataset procesado almacenado como carpeta.
    """

    name: str
    path: Path
    files: tuple[Path, ...]  # inventario de ficheros dentro


def list_processed_datasets(processed_root: Path) -> list[ProcessedDataset]:
    """
    TODO: Devuelve carpetas de datasets procesados dentro de processed_root.

    Cada dataset procesado es un directorio hijo: processed_root/<dataset_name>/
    """
    raise NotImplementedError("TODO: implement list_processed_datasets()")


def load_from_upload_xlsx(
    file_name: str, file_bytes: bytes, *, sheet: str | int | None = None
) -> LoadResult:
    """
    TODO:
    - Leer el XLSX (quizá seleccionar sheet)
    - Validar y preparar (limpieza, casting, etc.)
    - Devolver LoadResult(kind='upload')

    Nota: si sheet is None, decide una política (primera hoja / merge / error).
    """
    raise NotImplementedError("TODO: implement load_from_upload_xlsx()")


def list_xlsx_sheets(file_bytes: bytes) -> list[str]:
    """
    TODO: Devuelve los nombres de hojas disponibles en el xlsx subido.
    Útil para que la UI deje elegir hoja antes de cargar.
    """
    raise NotImplementedError("TODO: implement list_xlsx_sheets()")


def load_from_processed_dir(dataset_dir: Path) -> LoadResult:
    """
    TODO:
    - Cargar el 'df principal' desde un archivo dentro del directorio
      (por ejemplo: data.parquet o data.csv)
    - Devolver LoadResult(kind='processed', path=dataset_dir)
    """
    raise NotImplementedError("TODO: implement load_from_processed_dir()")


def get_processed_artifacts(dataset_dir: Path) -> dict[str, Path]:
    """
    TODO: Devuelve un mapping con artefactos relevantes:
      {
        "stats": <path a stats.json>,
        "graph": <path a graph.graphml>,
        "graph_meta": <path a graph_meta.json>,
        ...
      }
    La UI lo usará para mostrar "qué hay" sin conocer tu estructura exacta.
    """
    raise NotImplementedError("TODO: implement get_processed_artifacts()")


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """
    Placeholder de resumen de un DataFrame.

    Este método:
    - NO hace visualización
    - NO asume dominio
    - NO modifica el DataFrame
    - Devuelve solo información ligera y serializable

    La UI puede consumir esto directamente.
    """

    if df is None:
        raise ValueError("DataFrame is None")

    n_rows, n_cols = df.shape

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    summary: dict = {
        "shape": {
            "rows": int(n_rows),
            "columns": int(n_cols),
        },
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
        "missing_values": {
            "total": int(df.isna().sum().sum()),
            "by_column": df.isna().sum().to_dict(),
        },
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    return summary
