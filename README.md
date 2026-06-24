# GLAURLex

<p align="center">
  <img src="logo_glaur.png" alt="GLAURLex" width="260"/>
</p>

<p align="center">
  <em>Léxico disponible — análisis de disponibilidad léxica con grafos.</em>
</p>

GLAURLex is a small research app for **lexical-availability** studies
(*disponibilidad léxica*). It takes per-informant response lists collected
under a set of *temas* (semantic prompts), produces the usual descriptive
statistics, builds co-occurrence graphs per tema, and runs descriptive and
inferential analyses against informant metadata such as age, sex or level
of education.

The UI is in Spanish; the codebase is documented in English.

## What's in the box

- Ingest from **XLSX** spreadsheets or **Salamanca-format TXT** dumps.
- Per-type stats (frequency, availability, position, etc.) and per-informant
  summaries.
- Co-occurrence graphs (directed and undirected) per tema, with Leiden
  community detection (via `igraph` / `leidenalg`).
- Filter *grupos* over informants (sex, age bracket, education…) and rerun
  any analysis on the surviving subset.
- Inferential tests against informant variables, including ordinal
  variables you can order yourself.
- Persisted datasets on disk under `.data/processed/<dataset>/`, so heavy
  preprocessing only happens once.

## Quick start

Requires **Python 3.11** and Poetry.

```bash
poetry install
poetry run streamlit run src/glaurlex/ui/app.py
```

The UI is served at <http://localhost:8501>.

A few other things you can run:

```bash
poetry run pytest -q                  # tests
poetry run ruff check . && poetry run ruff format --check .
doxygen Doxyfile                      # API docs (Doxygen-style docstrings)
poetry run python scripts/run_glaurlex.py   # frozen-app entrypoint
```

## Repository layout

```
src/glaurlex/
  core/        Pure-Python domain layer: ingestion, graphs, stats, inference.
  ui/          Streamlit layer (views/, state, caching). Depends on core.
scripts/       Launchers and helper scripts (incl. PyInstaller entrypoint).
deploy/        Production stack: Traefik + Authelia (see below).
tests/         pytest suite.
```

The split between `core/` and `ui/` is enforced by convention — `core` never
imports Streamlit. Caching wrappers live in a single place
(`ui/metrics_cache.py`); metric metadata lives in `core/metrics_catalog.py`.
See [CLAUDE.md](CLAUDE.md) for a longer architecture note.

## Running with Docker

For a quick local container:

```bash
docker build -t glaurlex:latest .
docker run -p 8501:8501 glaurlex:latest
```

To exercise the full Traefik + Authelia stack locally (uses
`docker-compose.dev.yml`, served at `app.glaurlex.localhost` after adding
the hosts entries the script prints):

```bash
./scripts/dev-up.sh           # foreground; -d to detach
```

## Production deployment (`deploy/`)

The [`deploy/`](deploy/) folder contains the multi-tenant production stack:

- **Traefik** as reverse proxy with automatic Let's Encrypt certificates.
- **Authelia** as auth gate (forwardAuth), with argon2id-hashed users in
  `deploy/authelia/users_database.yml`.
- **GLAURLex** behind both, with per-user data sandboxes mounted at
  `${HOST_DATA_DIR}/<username>/processed`. The username comes from
  Authelia's `Remote-User` header and is sanitised before it touches the
  filesystem, so users can never see each other's datasets.

The full bring-up procedure (DNS, certs, secrets, user creation) lives in
[`deploy/README.md`](deploy/README.md).

## Desktop builds

The release workflow (`.github/workflows/release.yml`) produces Windows and
macOS PyInstaller artifacts on any `v*` tag. To reproduce locally:

```bash
poetry run pip install pyinstaller
poetry run pyinstaller --name GLAURLex --collect-all streamlit \
    --add-data "src/glaurlex/ui:glaurlex/ui" scripts/run_glaurlex.py
```

(Use `;` instead of `:` in the `--add-data` separator on Windows.)

## License

MIT — see [LICENSE](LICENSE).
