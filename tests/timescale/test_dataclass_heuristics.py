import dataclasses

from pydantic2django.django.timescale.heuristics import classify_dataclass_types, TimescaleRole


@dataclasses.dataclass
class ObsDC:
    device_id: str
    timestamp: int
    values: list[float]


def test_classify_dataclass_hypertable():
    roles = classify_dataclass_types([ObsDC])
    assert roles.get("ObsDC") == TimescaleRole.HYPERTABLE
