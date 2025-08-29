import pytest

from django.db import models
from pydantic import BaseModel

from pydantic2django.django.timescale.bases import PydanticTimescaleBase
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator


class SampleObservation(BaseModel):
    device_id: str
    timestamp: int
    value: float


def test_pydantic_generation_with_timescale_base(monkeypatch):
    # Use the real generator but force the base to be PydanticTimescaleBase to simulate hypertable path
    gen = StaticPydanticModelGenerator(output_path="/tmp/out.py", packages=["tests"])  # packages unused here
    # Force base for this run
    gen.base_model_class = PydanticTimescaleBase

    carrier = gen.setup_django_model(SampleObservation)
    assert carrier is not None
    assert carrier.django_model is not None
    # Ensure the generated dynamic model inherits from the Timescale-enabled base
    assert issubclass(carrier.django_model, PydanticTimescaleBase)
