"""! @package glaurlex.core.dataset_service
Servicios para procesar y cargar datasets en formato parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from glaurlex.core.graph import bigrams_for_tema


@dataclass(frozen=True)
class ProcessedDataset:
    """! Representa un dataset ya procesado (parquets en un directorio).

    Attributes:
        - `name`: Nombre del dataset.
        - `root`: Ruta al directorio raiz del dataset.
        - `informantes`: DataFrame con la tabla de informantes.
        - `temas`: Mapa tema -> DataFrame con columnas [user_id, pos, token].
    """

    name: str
    root: Path
    informantes: pd.DataFrame
    temas: Dict[str, pd.DataFrame]  # tema -> df con columnas [user_id, pos, token]


class DatasetService:
    """! Servicio de datasets.

    - procesa un XLSX a un directorio (parquets)
    - lista datasets ya procesados
    - carga un dataset procesado
    """

    def __init__(self, processed_dir: str | Path) -> None:
        """! Crea el servicio y asegura el directorio de salida.

        @param processed_dir Directorio base donde se guardan datasets procesados.
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def list_processed(self) -> List[str]:
        """! Lista nombres de datasets procesados.

        @return Lista de subdirectorios que contienen `informantes.parquet`.
        """
        out: List[str] = []
        for d in sorted(self.processed_dir.iterdir()):
            if d.is_dir() and (d / "informantes.parquet").exists():
                out.append(d.name)
        return out

    def get_processed_path(self, name: str) -> Path:
        """! Devuelve la ruta absoluta al dataset procesado.

        @param name Nombre del dataset.
        @return Ruta absoluta al directorio del dataset.
        """
        return (self.processed_dir / name).resolve()

    def is_processed(self, name: str) -> bool:
        """! Indica si el dataset ya está procesado.

        @param name Nombre del dataset.
        @return True si existe el directorio y `informantes.parquet`.
        """
        p = self.get_processed_path(name)
        return p.is_dir() and (p / "informantes.parquet").exists()

    def process_xlsx(
        self,
        xlsx_path: str | Path,
        dataset_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """! Procesa un XLSX y genera parquets en el directorio de procesados.

        @param xlsx_path Ruta al XLSX.
        @param dataset_name Nombre lógico del dataset (por defecto, nombre del archivo).
        @param overwrite Si True, reemplaza un dataset existente.
        @return Nombre del dataset procesado.
        @exception FileNotFoundError Si el XLSX no existe.
        @exception FileExistsError Si ya existe el dataset y overwrite es False.
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

        from glaurlex.core.xlsx_processing import pdprocessxlsx  # ajusta el path si cambia

        respath = str(out_dir) + "/"
        pdprocessxlsx(str(xlsx_path), respath)

        self._validate_processed_dir(out_dir)
        return dataset_name

    def process_salamanca(
        self,
        txt_path: str | Path,
        dataset_name: Optional[str] = None,
        overwrite: bool = False,
        stimulus_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """! Procesa un TXT formato Salamanca y genera parquets.

        @param txt_path Ruta al archivo .txt.
        @param dataset_name Nombre lógico del dataset (por defecto, nombre del archivo).
        @param overwrite Si True, reemplaza un dataset existente.
        @param stimulus_map Mapa opcional código -> estímulo.
        @return Nombre del dataset procesado.
        @exception FileNotFoundError Si el TXT no existe.
        @exception FileExistsError Si ya existe el dataset y overwrite es False.
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"No existe el TXT: {txt_path}")

        if dataset_name is None:
            dataset_name = txt_path.stem

        out_dir = self.get_processed_path(dataset_name)

        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Ya existe el dataset '{dataset_name}' en {out_dir}. "
                    f"Usa overwrite=True si quieres regenerarlo."
                )
        out_dir.mkdir(parents=True, exist_ok=True)

        from glaurlex.core.salamanca_processing import pdprocesssalamanca

        respath = str(out_dir) + "/"
        pdprocesssalamanca(str(txt_path), respath, stimulus_map=stimulus_map)

        self._validate_processed_dir(out_dir)
        return dataset_name

    def load_processed(self, name: str) -> ProcessedDataset:
        """! Carga informantes y todos los temas de un dataset procesado.

        @param name Nombre del dataset.
        @return Objeto ProcessedDataset con datos cargados.
        @exception FileNotFoundError Si el directorio no es válido.
        @exception ValueError Si falta el esquema esperado o no hay temas.
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
        """! Carga un único tema sin cargar todos.

        @param dataset_name Nombre del dataset.
        @param tema Nombre del tema (archivo parquet sin extensión).
        @return DataFrame del tema.
        @exception FileNotFoundError Si el tema no existe.
        """
        root = self.get_processed_path(dataset_name)
        self._validate_processed_dir(root)
        f = root / f"{tema}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"No existe el tema '{tema}' en {root}")
        return pd.read_parquet(f)

    def bigrams_for_tema(self, dataset_name: str, tema: str) -> pd.DataFrame:
        df = self.load_tema(dataset_name, tema)
        return bigrams_for_tema(df)

    def list_temas(self, dataset_name: str) -> List[str]:
        """! Lista los temas disponibles (parquets excepto informantes).

        @param dataset_name Nombre del dataset.
        @return Lista de nombres de temas.
        """
        root = self.get_processed_path(dataset_name)
        self._validate_processed_dir(root)
        temas = [f.stem for f in sorted(root.glob("*.parquet")) if f.name != "informantes.parquet"]
        return temas

    def _validate_processed_dir(self, root: Path) -> None:
        """! Valida la estructura mínima de un dataset procesado.

        @param root Ruta del directorio a validar.
        @exception FileNotFoundError Si falta el directorio o `informantes.parquet`.
        """
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"No existe el directorio del dataset: {root}")
        inf = root / "informantes.parquet"
        if not inf.exists():
            raise FileNotFoundError(f"Falta informantes.parquet en {root}")

    def load_informantes(self) -> pd.DataFrame:
        """! Carga el parquet de informantes del directorio base.

        @return DataFrame con informantes.
        @exception FileNotFoundError Si no existe `informantes.parquet`.
        """
        path = self.processed_dir / "informantes.parquet"
        if not path.exists():
            raise FileNotFoundError("No existe informantes.parquet")
        return pd.read_parquet(path)
