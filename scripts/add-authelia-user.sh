#!/usr/bin/env bash
# Añade un usuario a deploy/authelia/users_database.yml.
#
# Pregunta por username, email y password (silencioso), genera el hash
# argon2id con la imagen oficial de Authelia y añade el bloque del usuario
# al YAML. La *clave* del usuario se usa como nombre del sandbox de datos
# en <HOST_DATA_DIR>/<username>/processed, así que se restringe a
# [a-z0-9._-].
#
# Uso:
#   ./scripts/add-authelia-user.sh
#
# Tras añadirlo, recarga Authelia:
#   docker compose restart authelia

set -euo pipefail

cd "$(dirname "$0")/.."

DB="deploy/authelia/users_database.yml"
IMAGE="authelia/authelia:4.39.20"

if [[ ! -f "$DB" ]]; then
  echo "ERROR: no existe $DB" >&2
  exit 1
fi

read -rp "Username (clave del sandbox, [a-z0-9._-]): " username
if ! [[ "$username" =~ ^[a-z0-9._-]+$ ]]; then
  echo "ERROR: username inválido. Usa sólo [a-z0-9._-]." >&2
  exit 1
fi

if grep -qE "^[[:space:]]+${username}:[[:space:]]*$" "$DB"; then
  echo "ERROR: el usuario '${username}' ya existe en ${DB}." >&2
  exit 1
fi

read -rp "Email: " email

read -rsp "Password: " password; echo
read -rsp "Repite password: " password2; echo
if [[ -z "$password" ]]; then
  echo "ERROR: la contraseña no puede estar vacía." >&2
  exit 1
fi
if [[ "$password" != "$password2" ]]; then
  echo "ERROR: las contraseñas no coinciden." >&2
  exit 1
fi

echo "=> Generando hash argon2id con $IMAGE ..."
hash=$(docker run --rm "$IMAGE" \
  authelia crypto hash generate argon2 --password "$password" \
  | sed -n 's/^Digest: //p')

if [[ -z "$hash" ]]; then
  echo "ERROR: no se pudo generar el hash (¿está Docker disponible?)." >&2
  exit 1
fi

# Aseguramos newline final antes de añadir el bloque del nuevo usuario.
if [[ -n "$(tail -c1 "$DB")" ]]; then
  echo >> "$DB"
fi

cat >> "$DB" <<EOF
  ${username}:
    disabled: false
    displayname: "${username}"
    password: "${hash}"
    email: ${email}
    groups:
      - users
EOF

echo "=> Usuario '${username}' añadido a ${DB}."
echo "   Aplica los cambios con: docker compose restart authelia"
