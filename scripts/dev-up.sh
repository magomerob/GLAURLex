#!/usr/bin/env bash
# Bootstrap del stack local: genera secretos de Authelia, prepara el
# directorio de datos con el UID/GID correcto, y arranca docker compose.
#
# Pre-requisitos (una sola vez):
#   1. Añade a /etc/hosts:
#        127.0.0.1   app.glaurlex.localhost auth.glaurlex.localhost
#   2. Tener docker + docker compose y permisos para usar el socket.
#
# Uso:
#   ./scripts/dev-up.sh           # arranca en foreground (Ctrl-C para parar)
#   ./scripts/dev-up.sh -d        # detached
#
# Login: alice / changeme   o   bob / changeme

set -euo pipefail

cd "$(dirname "$0")/.."

PUID=${PUID:-$(id -u)}
PGID=${PGID:-$(id -g)}
export PUID PGID

if [[ "$PUID" == "0" || "$PGID" == "0" ]]; then
  echo "ERROR: PUID/PGID no pueden ser 0 (root). Exporta unos no-root antes de invocar." >&2
  exit 1
fi

mkdir -p deploy/authelia/secrets
for s in jwt_secret session_secret storage_encryption_key; do
  f="deploy/authelia/secrets/$s"
  if [[ ! -s "$f" ]]; then
    openssl rand -hex 32 | tr -d '\n' > "$f"
    chmod 600 "$f"
    echo "generado $f"
  fi
done

mkdir -p var/dev-data
# Aseguramos ownership coherente con el UID del contenedor. Si docker
# levantó el bind mount antes que nosotros, la carpeta puede haber quedado
# como root:root y el contenedor (non-root) no podría escribir en ella.
current_owner=$(stat -c '%u:%g' var/dev-data)
if [[ "$current_owner" != "${PUID}:${PGID}" ]]; then
  echo "=> var/dev-data es ${current_owner}, debería ser ${PUID}:${PGID}; corrigiendo..."
  if [[ "$(id -u)" == "0" ]]; then
    chown -R "${PUID}:${PGID}" var/dev-data
  elif sudo -n true 2>/dev/null; then
    sudo chown -R "${PUID}:${PGID}" var/dev-data
  else
    echo "   Necesito sudo para reasignar var/dev-data. Ejecuta:" >&2
    echo "     sudo chown -R ${PUID}:${PGID} var/dev-data" >&2
    exit 1
  fi
fi

echo
echo "=> /etc/hosts contiene:"
grep -E "app\.glaurlex\.localhost|auth\.glaurlex\.localhost" /etc/hosts || {
  echo "   (ninguna entrada) Añade como root:"
  echo "   127.0.0.1   app.glaurlex.localhost auth.glaurlex.localhost"
  echo
}

exec docker compose -f docker-compose.dev.yml up --build "$@"
