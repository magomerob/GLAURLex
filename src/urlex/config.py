import os
from pathlib import Path

APP_ENV = os.getenv("APP_ENV", "dev")

DEFAULT_PROCESSED_DIR = Path(".data/processed")
