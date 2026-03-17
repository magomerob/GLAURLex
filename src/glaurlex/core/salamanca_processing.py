"""! @package glaurlex.core.salamanca_processing
Placeholder para el procesamiento de archivos TXT (formato Salamanca) a parquets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*(.*?)\s*$")


def _sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\- ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "tema"


# FALTAN EL RESTO DE NIVELES
def _level_label_from_experiment(experiment_id: str) -> tuple[str, str]:
    if len(experiment_id) < 2:
        raise ValueError(f"Identificador de experimento inválido: '{experiment_id}'")
    level_code = experiment_id[-2:]
    level_map = {
        "12": "B1",
        "22": "C1",
    }
    return level_code, level_map.get(level_code, level_code)


def _resolve_stimulus_name(theme_code: str, stimulus_map: Optional[Dict[str, str]]) -> str:
    if not stimulus_map:
        return f"tema_{theme_code}"

    if theme_code in stimulus_map:
        return str(stimulus_map[theme_code])

    theme_code_no_zeros = theme_code.lstrip("0")
    if theme_code_no_zeros and theme_code_no_zeros in stimulus_map:
        return str(stimulus_map[theme_code_no_zeros])

    return f"tema_{theme_code}"


def pdprocesssalamanca(path, respath, stimulus_map: Optional[Dict[str, str]] = None):
    """! Procesa un TXT Salamanca y escribe parquets en el directorio de salida.

    @param path Ruta al archivo TXT de entrada.
    @param respath Directorio de salida para los parquets.
    @param stimulus_map Mapa opcional código -> estímulo.
    @exception ValueError Si el formato de alguna línea es inválido.
    """
    path = Path(path)
    respath = Path(respath)
    respath.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"No existe el TXT: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("El archivo Salamanca está vacío.")

    informantes_rows: list[dict] = []
    temas_rows: Dict[str, list[dict]] = {}

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        m = _LINE_RE.match(line)
        if m is None:
            raise ValueError(
                f"Línea {line_no}: formato inválido. Esperado: 'NNNNN NNN NN p1, p2, ...'"
            )

        experiment_id, individual_id_raw, theme_code, words_raw = m.groups()
        level_code, level_label = _level_label_from_experiment(experiment_id)
        individual_id = int(individual_id_raw)

        informantes_rows.append(
            {
                "CODIGO_INFORMANTE": individual_id,
                "EXPERIMENTO_ID": experiment_id,
                "NIVEL_CODIGO": level_code,
                "NIVEL": level_label,
            }
        )

        tokens = [tok.strip() for tok in words_raw.split(",")]
        # Eliminamos strings vacias
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            continue

        tema_name = _resolve_stimulus_name(theme_code, stimulus_map)
        temas_rows.setdefault(tema_name, [])
        for pos, token in enumerate(tokens):
            temas_rows[tema_name].append(
                {
                    "user_id": individual_id,
                    "pos": pos,
                    "type": token,
                }
            )

    if not informantes_rows:
        raise ValueError("No se encontraron líneas válidas en el archivo Salamanca.")

    if not temas_rows:
        raise ValueError("No se detectaron types para crear temas en el archivo Salamanca.")

    informantes_df = pd.DataFrame(informantes_rows)
    informantes_df = informantes_df.drop_duplicates(
        subset=["CODIGO_INFORMANTE", "EXPERIMENTO_ID", "NIVEL_CODIGO", "NIVEL"]
    )
    informantes_df = informantes_df.sort_values(
        ["CODIGO_INFORMANTE", "EXPERIMENTO_ID"]
    ).reset_index(drop=True)
    informantes_df.to_parquet(respath / "informantes.parquet", index=False)

    for tema_name, rows in temas_rows.items():
        df_tema = pd.DataFrame(rows, columns=["user_id", "pos", "type"])
        df_tema = df_tema.sort_values(["user_id", "pos"]).reset_index(drop=True)
        out_name = _sanitize_filename(tema_name) + ".parquet"
        df_tema.to_parquet(respath / out_name, index=False)
