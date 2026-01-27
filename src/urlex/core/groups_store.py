from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from urlex.core.groups import ALL_GROUP, Group


def groups_json_path(processed_dir: str, dataset_name: str) -> Path:
    return Path(processed_dir) / dataset_name / "groups.json"


def load_groups(processed_dir: str, dataset_name: str) -> Dict[str, Group]:
    path = groups_json_path(processed_dir, dataset_name)
    groups: Dict[str, Group] = {"TODOS": ALL_GROUP}

    if not path.exists():
        return groups

    data = json.loads(path.read_text(encoding="utf-8"))

    for item in data.get("groups", []):
        g = Group(
            name=item["name"],
            filters=item.get("filters", {}),
            immutable=bool(item.get("immutable", False)),
        )
        # no permitir sobrescribir TODOS ni crear inmutables raros desde disco
        if g.name.upper() == "TODOS":
            continue
        groups[g.name] = g

    # forzar TODOS siempre
    groups["TODOS"] = ALL_GROUP
    return groups


def save_groups(processed_dir: str, dataset_name: str, groups: Dict[str, Group]) -> None:
    path = groups_json_path(processed_dir, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardamos solo los no inmutables
    payload = {
        "dataset": dataset_name,
        "groups": [
            asdict(g) for g in groups.values() if not g.immutable and g.name.upper() != "TODOS"
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
