import pandas as pd

from glaurlex.core.groups import Group, apply_group


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ORIGEN": ["LOCAL", "HERENCIA", "OTRO", "LOCAL", "HERENCIA"],
            "NIVELOPT": [1, 2, 2, 3, 2],
            "EDAD": [25, 29, 25, 31, 22],
            "CIUDAD": ["MADRID", "MADRID", "MADRID", "MADRID", "MADRID"],
            "GENERO": ["F", "M", "F", "F", "F"],
        }
    )


def test_apply_group_combined_rules() -> None:
    df = _sample_df()
    group = Group(
        name="FILTRO",
        filters={
            "ORIGEN": ["LOCAL", "HERENCIA"],
            "NIVELOPT": {">=": 2},
            "EDAD": {">=": 20, "<=": 30},
            "CIUDAD": "MADRID",
            "GENERO": {"!=": "M"},
        },
    )

    filtered = apply_group(df, group)

    assert len(filtered) == 1
    row = filtered.iloc[0]
    assert row["ORIGEN"] == "HERENCIA"
    assert row["NIVELOPT"] == 2
    assert row["EDAD"] == 22
    assert row["CIUDAD"] == "MADRID"
    assert row["GENERO"] == "F"


def test_apply_group_scalar_rule() -> None:
    df = _sample_df()
    group = Group(name="LOCAL", filters={"ORIGEN": "LOCAL"})

    filtered = apply_group(df, group)

    assert filtered["ORIGEN"].tolist() == ["LOCAL", "LOCAL"]
