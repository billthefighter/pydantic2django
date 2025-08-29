#!/usr/bin/env bash
set -euo pipefail

echo "🐘 Ensuring TimescaleDB container is running on localhost:6543..."
if ! docker container inspect timescaledb >/dev/null 2>&1; then \
  echo "➡️  Creating and starting new container 'timescaledb'"; \
  docker run -d --name timescaledb -p 6543:5432 \
    -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=p2ddb \
    timescale/timescaledb:latest-pg17 >/dev/null; \
else \
  if [ -z "$(docker ps --filter name=^/timescaledb$ --filter status=running --quiet)" ]; then \
    echo "➡️  Starting existing container 'timescaledb'"; \
    docker start timescaledb >/dev/null; \
  else \
    echo "✅ TimescaleDB container already running"; \
  fi; \
fi
echo "⏳ Waiting for database to become ready..."
for i in $(seq 1 60); do \
  if docker exec timescaledb pg_isready -h 127.0.0.1 -p 5432 -U postgres -d p2ddb >/dev/null 2>&1; then \
    echo "✅ TimescaleDB is ready"; \
    break; \
  fi; \
  sleep 1; \
done; \
if ! docker exec timescaledb pg_isready -h 127.0.0.1 -p 5432 -U postgres >/dev/null 2>&1; then \
  echo "❌ TimescaleDB did not become ready in time"; exit 1; \
fi

# Ensure database p2ddb exists (container may predate env var)
if ! docker exec -u postgres timescaledb psql -Atqc "SELECT 1 FROM pg_database WHERE datname='p2ddb'" | grep -q 1; then \
  echo "🆕 Creating database p2ddb"; \
  docker exec -u postgres timescaledb createdb -O postgres p2ddb; \
fi

echo "✅ Database p2ddb is present"

# Ensure TimescaleDB extension is enabled in p2ddb
docker exec -u postgres timescaledb psql -d p2ddb -qc "CREATE EXTENSION IF NOT EXISTS timescaledb;" >/dev/null 2>&1 || true
echo "✅ TimescaleDB extension ensured in p2ddb"
