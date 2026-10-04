#!/usr/bin/env bash
# Start the isolated Langfuse stack (UI http://localhost:3100). Secrets are generated once into docker/langfuse/.env.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  r() { openssl rand -hex 16; }
  { echo "PG_PASS=$(r)"; echo "CH_PASS=$(r)"; echo "REDIS_PASS=$(r)"; echo "MINIO_PASS=$(r)"; echo "SALT=$(r)"
    echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"; echo "NEXTAUTH_SECRET=$(r)"; echo "LF_USER_PASS=$(r)"
    echo "LF_PUBLIC=pk-lf-$(r)"; echo "LF_SECRET=sk-lf-$(r)"; } > .env
  chmod 600 .env
fi
docker compose --env-file .env up -d
