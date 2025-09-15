import time
from pathlib import Path

import pytest

from pydantic2django.xmlschema.ingestor import (
    clear_ingestor_cache,
    get_shared_ingestor,
    ingestor_cache_stats,
    set_ingestor_cache,
)


def _fixture_xsd_root() -> Path:
    here = Path(__file__).resolve().parent
    # Reuse built-in simple schema for fast tests (tests/xmlschema/fixtures/...)
    return (here / "fixtures" / "simple_schema.xsd").resolve()


@pytest.fixture(autouse=True)
def reset_cache():
    clear_ingestor_cache()
    set_ingestor_cache(maxsize=8, ttl_seconds=60)
    yield
    clear_ingestor_cache()


def test_shared_ingestor_reuse_within_ttl():
    xsd = str(_fixture_xsd_root())
    app = "tests"

    a = get_shared_ingestor(schema_files=[xsd], app_label=app)
    b = get_shared_ingestor(schema_files=[xsd], app_label=app)

    assert a is b, "Expected same shared ingestor instance within TTL"


def test_shared_ingestor_lru_eviction(tmp_path: Path):
    xsd = str(_fixture_xsd_root())
    # Tight LRU
    set_ingestor_cache(maxsize=2, ttl_seconds=600)

    a = get_shared_ingestor(schema_files=[xsd], app_label="tests-a")
    b = get_shared_ingestor(schema_files=[xsd], app_label="tests-b")
    # Access a to make b LRU
    _ = get_shared_ingestor(schema_files=[xsd], app_label="tests-a")
    _ = get_shared_ingestor(schema_files=[xsd], app_label="tests-c")

    # Since maxsize=2 and a was most-recently-used at the time of inserting c,
    # b should be evicted. Fetching b again will reinsert it and may evict 'a'.
    b2 = get_shared_ingestor(schema_files=[xsd], app_label="tests-b")

    assert b2 is not b, "Least recently used entry should have been evicted"


def test_shared_ingestor_ttl_expiry(monkeypatch):
    xsd = str(_fixture_xsd_root())

    # Simulate time progression by monkeypatching time.time used inside cache
    import pydantic2django.xmlschema.ingestor as ing

    base = time.time()
    t = {"now": base}

    def fake_time():
        return t["now"]

    # Recreate cache with short TTL and fake clock
    ing._INGESTOR_CACHE = ing._LruTtlCache(maxsize=8, ttl_seconds=0.05, now=fake_time)

    a = get_shared_ingestor(schema_files=[xsd], app_label="tests")
    # Advance beyond TTL
    t["now"] = base + 0.10
    b = get_shared_ingestor(schema_files=[xsd], app_label="tests")

    assert a is not b, "Entry should expire after TTL"
