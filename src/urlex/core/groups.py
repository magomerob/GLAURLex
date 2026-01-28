"""! @package urlex.core.groups
Definición y aplicación de grupos de filtros.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Group:
    """! Define un grupo de filtrado sobre informantes.

    Attributes:
        - `name`: Nombre del grupo.
        - `filters`: Mapa columna -> regla de filtrado.
        - `immutable`: Si True, no debe persistirse ni sobrescribirse.
    """

    name: str
    filters: Dict[str, Any]  # ej: {"ORIGEN": ["LOCAL", "HERENCIA"]}
    immutable: bool = False


ALL_GROUP = Group(name="TODOS", filters={}, immutable=True)


def apply_group(df_informantes, group: Group):
    """! Aplica el filtro de un grupo sobre un DataFrame de informantes.

    @param df_informantes DataFrame con datos de informantes.
    @param group Grupo de filtrado a aplicar.
    @return DataFrame filtrado.
    """
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
