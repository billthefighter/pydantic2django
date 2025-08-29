from pydantic import BaseModel

from pydantic2django.django.timescale.heuristics import classify_pydantic_models, TimescaleRole


class Obs(BaseModel):
    device_id: str
    timestamp: int
    values: list[float]


def test_classify_pydantic_models_hypertable():
    roles = classify_pydantic_models([Obs])
    assert roles.get("Obs") == TimescaleRole.HYPERTABLE
