"""! @package glaurlex.core.example_data
Genera datasets de ejemplo (XLSX y Salamanca TXT) para descargar desde la UI.

Los ejemplos se construyen en memoria, con la misma estructura que espera
`xlsx_processing.pdprocessxlsx` y `salamanca_processing.pdprocesssalamanca`,
de modo que un usuario pueda descargarlos, inspeccionarlos y volver a
procesarlos como plantilla. No se guardan ficheros binarios en el repositorio:
esto evita problemas de empaquetado en la app congelada (PyInstaller) y en
Docker.
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pandas as pd

EXAMPLE_XLSX_FILENAME = "ejemplo_glaurlex.xlsx"
EXAMPLE_SALAMANCA_FILENAME = "ejemplo_salamanca.txt"

# --- Metadatos de informantes (compartidos por ambos ejemplos) --------------

# Cada variable define sus etiquetas en orden: el código 1 corresponde a la
# primera etiqueta, el 2 a la segunda, etc. (igual que la hoja "Variables").
_VARIABLE_LABELS: Dict[str, List[str]] = {
    "SEXO": ["HOMBRE", "MUJER"],
    "NIVEL": ["A1", "A2", "B1", "B2", "C1"],
    "LENGUAMATERNA": ["ESPAÑOL", "INGLES", "ARABE", "CHINO", "OTRO"],
}

# Un informante por fila: (SEXO, NIVEL, LENGUAMATERNA) como códigos 1-based.
_INFORMANTE_CODES: List[tuple[int, int, int]] = [
    (1, 3, 2),
    (2, 3, 1),
    (1, 4, 3),
    (2, 4, 2),
    (1, 5, 1),
    (2, 5, 4),
    (1, 3, 5),
    (2, 4, 1),
    (1, 5, 2),
    (2, 3, 3),
    (1, 4, 4),
    (2, 5, 1),
    (1, 3, 1),
    (2, 4, 5),
    (1, 5, 3),
    (2, 3, 2),
]

# --- Respuestas por tema (una lista de palabras por informante) -------------
#
# El orden importa: la posición de cada palabra es su "rango" de disponibilidad
# léxica. El vocabulario se solapa entre informantes a propósito para que los
# grafos de coocurrencia (bigramas) tengan aristas con peso > 1.

_THEMES: Dict[str, List[List[str]]] = {
    "Animales": [
        ["perro", "gato", "caballo", "vaca", "oveja", "cerdo"],
        ["gato", "perro", "raton", "conejo", "pajaro"],
        ["leon", "tigre", "elefante", "jirafa", "mono", "cebra"],
        ["perro", "gato", "pez", "tortuga", "hamster"],
        ["vaca", "caballo", "cerdo", "gallina", "oveja", "pato"],
        ["gato", "perro", "conejo", "raton", "pajaro", "pez"],
        ["leon", "elefante", "tigre", "oso", "lobo", "zorro"],
        ["perro", "gato", "caballo", "conejo", "tortuga"],
        ["vaca", "oveja", "cerdo", "gallina", "caballo"],
        ["mono", "jirafa", "leon", "tigre", "cebra", "elefante"],
        ["gato", "perro", "pajaro", "pez", "hamster", "conejo"],
        ["lobo", "zorro", "oso", "leon", "tigre"],
        ["perro", "gato", "vaca", "caballo", "gallina", "pato"],
        ["raton", "conejo", "pajaro", "gato", "perro"],
        ["elefante", "jirafa", "mono", "cebra", "leon"],
        ["caballo", "vaca", "oveja", "cerdo", "perro", "gato"],
    ],
    "Comida_Bebida": [
        ["agua", "pan", "leche", "cafe", "fruta", "carne"],
        ["pan", "queso", "jamon", "agua", "vino"],
        ["arroz", "pasta", "pollo", "pescado", "verdura"],
        ["cafe", "leche", "azucar", "pan", "mantequilla"],
        ["agua", "cerveza", "vino", "refresco", "zumo"],
        ["pan", "queso", "tomate", "aceite", "sal"],
        ["pollo", "arroz", "pescado", "carne", "verdura", "fruta"],
        ["cafe", "te", "leche", "agua", "zumo"],
        ["pan", "jamon", "queso", "huevo", "aceite"],
        ["manzana", "platano", "naranja", "fruta", "agua"],
        ["pasta", "arroz", "pollo", "tomate", "queso"],
        ["cerveza", "vino", "agua", "refresco", "cafe"],
        ["pan", "leche", "cafe", "azucar", "mantequilla", "fruta"],
        ["pescado", "carne", "pollo", "arroz", "verdura"],
        ["agua", "zumo", "leche", "cafe", "te"],
        ["tomate", "aceite", "sal", "pan", "queso", "huevo"],
    ],
    "Ciudad": [
        ["calle", "plaza", "parque", "tienda", "coche", "casa"],
        ["edificio", "calle", "semaforo", "acera", "farola"],
        ["mercado", "tienda", "restaurante", "bar", "cafeteria"],
        ["parque", "plaza", "fuente", "banco", "arbol"],
        ["coche", "autobus", "metro", "bicicleta", "taxi"],
        ["hospital", "colegio", "iglesia", "ayuntamiento", "biblioteca"],
        ["calle", "plaza", "parque", "mercado", "tienda", "bar"],
        ["edificio", "casa", "piso", "portal", "ascensor"],
        ["semaforo", "acera", "calle", "coche", "autobus"],
        ["restaurante", "bar", "cafeteria", "tienda", "mercado"],
        ["parque", "arbol", "banco", "fuente", "plaza"],
        ["metro", "autobus", "tren", "estacion", "taxi"],
        ["colegio", "hospital", "biblioteca", "iglesia", "museo"],
        ["calle", "farola", "semaforo", "acera", "edificio"],
        ["tienda", "mercado", "supermercado", "panaderia", "farmacia"],
        ["plaza", "parque", "fuente", "estatua", "banco", "arbol"],
    ],
}


def _build_informantes_df() -> pd.DataFrame:
    rows = []
    for i, (sexo, nivel, lengua) in enumerate(_INFORMANTE_CODES, start=1):
        rows.append(
            {
                "CODIGO_INFORMANTE": i,
                "SEXO": sexo,
                "NIVEL": nivel,
                "LENGUAMATERNA": lengua,
            }
        )
    return pd.DataFrame(rows)


def _build_variables_df() -> pd.DataFrame:
    # Columnas de longitud desigual: se rellenan con NaN al final, igual que en
    # los ficheros reales.
    return pd.DataFrame({name: pd.Series(labels) for name, labels in _VARIABLE_LABELS.items()})


def build_example_xlsx() -> bytes:
    """! Construye el XLSX de ejemplo en memoria.

    Estructura:
      - Hoja `Informantes`: código + variables (códigos 1-based).
      - Hoja `Variables`: etiquetas de cada variable, una por fila.
      - Una hoja por tema con la **primera fila vacía** (cabecera en blanco) y,
        a partir de la segunda, una fila por informante con sus palabras en
        orden de disponibilidad.

    @return Contenido del XLSX como bytes.
    """
    n_inf = len(_INFORMANTE_CODES)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _build_informantes_df().to_excel(writer, sheet_name="Informantes", index=False)
        _build_variables_df().to_excel(writer, sheet_name="Variables", index=False)

        for tema, respuestas in _THEMES.items():
            # Una fila por informante (rellenamos si hay menos respuestas que
            # informantes para mantener la correspondencia posicional).
            filas = list(respuestas[:n_inf])
            while len(filas) < n_inf:
                filas.append([])
            df_tema = pd.DataFrame(filas)
            # `startrow=1` + `header=False` deja la primera fila del Excel en
            # blanco: al releer con pandas la cabecera queda como "Unnamed: N",
            # tal y como esperan los ficheros reales.
            df_tema.to_excel(
                writer,
                sheet_name=tema,
                index=False,
                header=False,
                startrow=1,
            )

    return buffer.getvalue()


def build_example_salamanca() -> str:
    """! Construye el TXT de ejemplo en formato Salamanca.

    Cada línea tiene el formato `EXPID INDIV TEMA palabra1, palabra2, ...`.
    Los dos últimos dígitos de `EXPID` codifican el nivel (`12` -> B1,
    `22` -> C1); aquí la primera mitad de informantes es B1 y la segunda C1.
    Los códigos de tema (`01`, `02`, `03`) se corresponden con los temas del
    XLSX de ejemplo, por lo que pueden nombrarse con un diccionario
    `codigo=estimulo` al procesar.

    @return Texto del fichero Salamanca.
    """
    n_inf = len(_INFORMANTE_CODES)
    half = n_inf // 2
    tema_codes = {tema: f"{i + 1:02d}" for i, tema in enumerate(_THEMES)}

    lines: List[str] = []
    for tema, respuestas in _THEMES.items():
        code = tema_codes[tema]
        for idx in range(n_inf):
            exp_id = "50012" if idx < half else "50022"
            individual_id = idx + 1
            palabras = respuestas[idx] if idx < len(respuestas) else []
            if not palabras:
                continue
            lines.append(f"{exp_id} {individual_id} {code} {', '.join(palabras)}")

    return "\n".join(lines) + "\n"


def salamanca_stimulus_map() -> Dict[str, str]:
    """! Diccionario `codigo -> estimulo` que acompaña al TXT de ejemplo.

    Útil para mostrarlo en la UI como sugerencia al procesar el ejemplo
    Salamanca, de modo que los temas se nombren en lugar de quedar como
    `tema_01`, `tema_02`, ...

    @return Mapa de código de tema a nombre de estímulo.
    """
    return {f"{i + 1:02d}": tema for i, tema in enumerate(_THEMES)}
