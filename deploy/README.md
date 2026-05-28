# Despliegue: Traefik + Authelia + GLAURLex

Stack `docker compose` para servir GLAURLex en un servidor detrás de Traefik
(reverse proxy + Let's Encrypt) con Authelia haciendo de gate de
autenticación. Cada usuario autenticado obtiene su propio sandbox de datos
bajo `<HOST_DATA_DIR>/<username>/processed`.

## Pre-requisitos

- Un servidor con Docker y `docker compose`.
- DNS apuntando `APP_HOST` y `AUTH_HOST` a la IP del servidor (puertos 80
  y 443 abiertos hacia fuera para que ACME funcione).
- Un usuario en el host con UID/GID conocidos para correr el contenedor
  (NO root). Por defecto `1000:1000`.

## Primera vez

1. **Configura `.env`**
   ```bash
   cp .env.example .env
   $EDITOR .env
   ```

2. **Prepara el almacén de certificados** (debe existir y ser legible sólo
   por root porque Traefik corre como root y comprueba los permisos):
   ```bash
   touch deploy/traefik/acme.json
   chmod 600 deploy/traefik/acme.json
   ```

3. **Genera los secretos de Authelia** (uno por archivo, sin newline):
   ```bash
   mkdir -p deploy/authelia/secrets
   for s in jwt_secret session_secret storage_encryption_key; do
     openssl rand -hex 32 | tr -d '\n' > deploy/authelia/secrets/$s
   done
   chmod 600 deploy/authelia/secrets/*
   ```

4. **Crea usuarios en Authelia.** Genera el hash argon2id y pégalo en
   `deploy/authelia/users_database.yml`:
   ```bash
   docker run --rm authelia/authelia:4.39.20 \
     authelia crypto hash generate argon2 -- 'TU_PASSWORD'
   ```
   La *clave* del usuario en ese YAML (`alice`, `bob`, ...) se usará como
   nombre del sandbox de datos en `<HOST_DATA_DIR>/<clave>/processed`.
   Restringe la clave a `[a-z0-9._-]`.

5. **Prepara el directorio de datos en el host** con el UID/GID correctos:
   ```bash
   mkdir -p ./var/data        # ajustado a HOST_DATA_DIR en .env
   sudo chown -R 1000:1000 ./var/data
   ```

6. **Arranca**:
   ```bash
   docker compose up -d --build
   docker compose logs -f traefik authelia glaurlex
   ```

Al abrir `https://${APP_HOST}` Traefik te redirige al portal de Authelia
en `https://${AUTH_HOST}`. Tras login, vuelves a la app con tu sesión y
sólo verás los datasets de tu sandbox.

## Cómo funciona el aislamiento por usuario

1. Authelia valida la sesión y, al responder al `forwardAuth` de Traefik,
   añade la cabecera `Remote-User: <username>`.
2. Traefik propaga esa cabecera al contenedor de la app.
3. La app la lee en `glaurlex.ui.state._remote_username()`, la sanea
   (regex `[A-Za-z0-9._-]`) y fija `processed_dir` en sesión a
   `${GLAURLEX_DATA_DIR}/<username>/processed`. El campo "Directorio de
   datasets procesados" de la pantalla de carga queda bloqueado, así que
   un usuario no puede escapar a las carpetas de otro.
4. Si la cabecera no llega (p. ej. saltándose el proxy) y
   `GLAURLEX_REQUIRE_AUTH=1`, la app detiene la página con un mensaje
   de acceso denegado.

## Mantenimiento

- **Añadir usuario**: edita `deploy/authelia/users_database.yml` y reinicia
  `docker compose restart authelia`. El sandbox `<HOST_DATA_DIR>/<user>/processed`
  se crea automáticamente al primer login.
- **Renovación TLS**: la maneja Traefik automáticamente.
- **Backups**: basta con copiar `HOST_DATA_DIR` y el volumen `authelia_data`
  (sesiones/regulación).
- **Rotar secretos de Authelia**: regenera los ficheros en
  `deploy/authelia/secrets/` y reinicia Authelia (invalida sesiones).
