from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Group:
    name: str
    filters: Dict[str, Any]  # ej: {"ORIGEN": ["LOCAL", "HERENCIA"]}
    immutable: bool = False


ALL_GROUP = Group(name="TODOS", filters={}, immutable=True)


def apply_group(df_informantes, group: Group):
    df = df_informantes.copy()
    for col, rule in group.filters.items():
        if isinstance(rule, list):
            df = df[df[col].isin(rule)]
        elif isinstance(rule, dict):
            for op, val in rule.items():
                if op == "<=":
                    df = df[df[col] <= val]
                elif op == ">=":
                    df = df[df[col] >= val]
                elif op == "!=":
                    df = df[df[col] != val]
                elif op == "==":
                    df = df[df[col] == val]
        else:
            df = df[df[col] == rule]
    return df
