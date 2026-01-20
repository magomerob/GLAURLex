from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class ProcessedDataset:
    """Representa un dataset ya procesado (parquets en un directorio)."""

    name: str
    root: Path
    informantes: pd.DataFrame
    temas: Dict[str, pd.DataFrame]  # tema -> df con columnas [user_id, pos, token]


class DatasetService:
    """
    Servicio de datasets:
      - procesa un XLSX a un directorio (parquets)
      - lista datasets ya procesados
      - carga un dataset procesado
    """

    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def list_processed(self) -> List[str]:
        """Lista nombres de datasets procesados (subdirectorios con informantes.parquet)."""
        out: List[str] = []
        for d in sorted(self.processed_dir.iterdir()):
            if d.is_dir() and (d / "informantes.parquet").exists():
                out.append(d.name)
        return out

    def get_processed_path(self, name: str) -> Path:
        return (self.processed_dir / name).resolve()

    def is_processed(self, name: str) -> bool:
        p = self.get_processed_path(name)
        return p.is_dir() and (p / "informantes.parquet").exists()

    def process_xlsx(
        self,
        xlsx_path: str | Path,
        dataset_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Procesa un XLSX usando pdprocessxlsx y deja el resultado en processed_dir/dataset_name/.

        Devuelve el nombre del dataset procesado.
        """
        xlsx_path = Path(xlsx_path)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"No existe el XLSX: {xlsx_path}")

        if dataset_name is None:
            dataset_name = xlsx_path.stem

        out_dir = self.get_processed_path(dataset_name)

        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Ya existe el dataset '{dataset_name}' en {out_dir}. "
                    f"Usa overwrite=True si quieres regenerarlo."
                )
        out_dir.mkdir(parents=True, exist_ok=True)

        from urlex.core.xlsx_processing import pdprocessxlsx  # ajusta el path si cambia

        respath = str(out_dir) + "/"
        pdprocessxlsx(str(xlsx_path), respath)

        self._validate_processed_dir(out_dir)
        return dataset_name

    def load_processed(self, name: str) -> ProcessedDataset:
        """
        Carga informantes + todos los temas de un dataset procesado.
        """
        root = self.get_processed_path(name)
        self._validate_processed_dir(root)

        informantes = pd.read_parquet(root / "informantes.parquet")

        temas: Dict[str, pd.DataFrame] = {}
        for f in sorted(root.glob("*.parquet")):
            if f.name == "informantes.parquet":
                continue
            tema_name = f.stem
            df = pd.read_parquet(f)

            # Aseguramos el esquema esperado
            required = {"user_id", "pos", "token"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"El tema '{tema_name}' no tiene columnas {sorted(required)}. "
                    f"Faltan: {sorted(missing)} en {f}"
                )

            # Normalización ligera
            df = df[["user_id", "pos", "token"]].copy()
            temas[tema_name] = df

        if not temas:
            raise ValueError(
                f"Dataset '{name}' no tiene temas (no hay parquets aparte de informantes)."
            )

        return ProcessedDataset(
            name=name,
            root=root,
            informantes=informantes,
            temas=temas,
        )

    def load_tema(self, dataset_name: str, tema: str) -> pd.DataFrame:
        """Carga un único tema sin cargar todos."""
        root = self.get_processed_path(dataset_name)
        self._validate_processed_dir(root)
        f = root / f"{tema}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"No existe el tema '{tema}' en {root}")
        return pd.read_parquet(f)

    def list_temas(self, dataset_name: str) -> List[str]:
        """Lista los temas disponibles (parquets excepto informantes)."""
        root = self.get_processed_path(dataset_name)
        self._validate_processed_dir(root)
        temas = [f.stem for f in sorted(root.glob("*.parquet")) if f.name != "informantes.parquet"]
        return temas

    def _validate_processed_dir(self, root: Path) -> None:
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"No existe el directorio del dataset: {root}")
        inf = root / "informantes.parquet"
        if not inf.exists():
            raise FileNotFoundError(f"Falta informantes.parquet en {root}")
