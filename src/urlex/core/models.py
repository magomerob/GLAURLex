"""! @package urlex.core.models
Modelos base del módulo core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

SourceKind = Literal["upload", "processed"]


@dataclass(frozen=True)
class DatasetRef:
    """! Referencia a un dataset.

    Attributes:
        - `kind`: Tipo de origen (upload/processed).
        - `name`: Nombre del dataset.
        - `path`: Ruta opcional al directorio del dataset procesado.
    """

    kind: SourceKind
    name: str
    path: Optional[Path] = None  # si processed => carpeta dataset


@dataclass(frozen=True)
class LoadResult:
    """! Resultado de una carga de dataset.

    Attributes:
        - `ref`: Referencia al dataset.
        - `df`: DataFrame cargado.
    """

    ref: DatasetRef
    df: pd.DataFrame
