import sys

from streamlit.web import cli as stcli


def main() -> int:
    sys.argv = ["streamlit", "run", "src/urlex/ui/app.py"]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
