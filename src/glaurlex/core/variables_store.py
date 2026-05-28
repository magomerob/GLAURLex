"""! @package glaurlex.core.variables_store
Persistencia de metadatos de variables (p. ej. ordinales) por dataset procesado.

Estructura de `variables.json`:

```
{
  "dataset": "salamanca",
  "variables": {
    "NIVEL_ESTUDIOS": {"ordinal": true, "order": ["primaria", "secundaria", "universitaria"]},
    "EDAD_GRUPO":     {"ordinal": true, "order": ["joven", "adulto", "mayor"]}
  }
}
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def variables_json_path(processed_dir: str, dataset_name: str) -> Path:
    """! Ruta al archivo `variables.json` de un dataset.

    @param processed_dir Directorio base de datasets procesados.
    @param dataset_name Nombre del dataset.
    @return Ruta absoluta al `variables.json`.
    """
    return Path(processed_dir) / dataset_name / "variables.json"


def load_variables(processed_dir: str, dataset_name: str) -> Dict[str, Dict]:
    """! Carga la configuración de variables del dataset.

    @return Diccionario `col -> {"ordinal": bool, "order": list[str]}`.
        Devuelve `{}` si el archivo no existe.
    """
    path = variables_json_path(processed_dir, dataset_name)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("variables", {}) or {}
    out: Dict[str, Dict] = {}
    for col, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        ordinal = bool(cfg.get("ordinal", False))
        order = cfg.get("order", []) or []
        if not isinstance(order, list):
            order = []
        out[col] = {"ordinal": ordinal, "order": [str(x) for x in order]}
    return out


def save_variables(
    processed_dir: str, dataset_name: str, config: Dict[str, Dict]
) -> None:
    """! Guarda la configuración de variables.

    Solo persiste columnas que tengan `ordinal=True` y al menos un nivel
    declarado en `order`.
    """
    path = variables_json_path(processed_dir, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned: Dict[str, Dict] = {}
    for col, cfg in config.items():
        if not isinstance(cfg, dict):
            continue
        if not cfg.get("ordinal"):
            continue
        order: List[str] = [str(x) for x in (cfg.get("order") or [])]
        if not order:
            continue
        cleaned[col] = {"ordinal": True, "order": order}

    payload = {"dataset": dataset_name, "variables": cleaned}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_order(config: Dict[str, Dict], col: str) -> List[str]:
    """! Helper: devuelve la lista de niveles ordenados, o `[]` si no es ordinal."""
    cfg = config.get(col) or {}
    if not cfg.get("ordinal"):
        return []
    return list(cfg.get("order") or [])


def is_ordinal(config: Dict[str, Dict], col: str) -> bool:
    """! Helper: True si la columna está marcada como ordinal con orden válido."""
    return bool(get_order(config, col))
