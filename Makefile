.PHONY: timescaledb-up test-timescale-integration

timescaledb-up:
	@bash scripts/start_timescaledb.sh

test-timescale-integration: timescaledb-up
	@echo "🔧 Setting env and running integration test..."
	@export DB_HOST=127.0.0.1 DB_PORT=6543 DB_NAME=p2ddb DB_USER=postgres DB_PASSWORD=postgres P2D_INTEGRATION_DB=1; \
	uv run pytest -q tests/integration/test_timescale_fk.py -q
