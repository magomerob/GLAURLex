from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

SourceKind = Literal["upload", "processed"]


@dataclass(frozen=True)
class DatasetRef:
    kind: SourceKind
    name: str
    path: Optional[Path] = None  # si processed => carpeta dataset


@dataclass(frozen=True)
class LoadResult:
    ref: DatasetRef
    df: pd.DataFrame
