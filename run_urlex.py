import os
import sys

from streamlit.web import cli as stcli

from urlex.ui import app as urlex_app


def main() -> int:
    # Detecta pyinstaller
    if hasattr(sys, "_MEIPASS"):
        app_path = os.path.join(sys._MEIPASS, "urlex", "ui", "app.py")
    else:
        app_path = urlex_app.__file__
    sys.argv = ["streamlit", "run", app_path]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
