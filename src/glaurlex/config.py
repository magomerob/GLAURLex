"""! @package glaurlex.config
Configuración global derivada de variables de entorno.

Variables soportadas:
    - `GLAURLEX_DATA_DIR`: raíz de datos. En despliegues multiusuario es la
      base bajo la cual se crean subdirectorios por usuario
      (`<GLAURLEX_DATA_DIR>/<username>/processed`). En local, por defecto
      se usa `.data/processed` directamente y no se aplica scoping.
    - `GLAURLEX_REMOTE_USER_HEADER`: nombre de la cabecera HTTP que el
      proxy (Authelia/Traefik) inyecta con el usuario autenticado.
    - `GLAURLEX_REQUIRE_AUTH`: si es `1`/`true`, la app exige que la
      cabecera de usuario esté presente y rechaza el acceso anónimo.
    - `GLAURLEX_LOGOUT_URL`: URL a la que redirige el botón de cierre de
      sesión (p. ej. el endpoint `/logout` de Authelia). Si está vacía no
      se muestra el botón.
    - `APP_ENV`: etiqueta de entorno (`dev` por defecto).
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ENV = os.getenv("APP_ENV", "dev")


def _data_root() -> Path:
    raw = os.getenv("GLAURLEX_DATA_DIR")
    if raw:
        return Path(raw)
    return Path(".data/processed")


DATA_ROOT = _data_root()

DEFAULT_PROCESSED_DIR = DATA_ROOT

REMOTE_USER_HEADER = os.getenv("GLAURLEX_REMOTE_USER_HEADER", "Remote-User")
REQUIRE_AUTH = os.getenv("GLAURLEX_REQUIRE_AUTH", "").lower() in {"1", "true", "yes"}
LOGOUT_URL = os.getenv("GLAURLEX_LOGOUT_URL", "").strip()


def user_processed_dir(username: str) -> Path:
    """! Construye el directorio procesado aislado por usuario.

    @param username Nombre de usuario ya saneado.
    @return `<DATA_ROOT>/<username>/processed`.
    """
    return DATA_ROOT / username / "processed"
