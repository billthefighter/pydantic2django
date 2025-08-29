import dataclasses

from django.db import models

from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator
from pydantic2django.django.timescale.bases import DataclassTimescaleBase


@dataclasses.dataclass
class AutoObsDC:
    device_id: str
    timestamp: int
    values: list[float]


def test_dataclass_generator_picks_timescale_base_automatically():
    gen = DataclassDjangoModelGenerator(
        output_path="/tmp/out.py",
        app_label="tests",
        filter_function=None,
        verbose=False,
        packages=["tests"],
    )
    carrier = gen.setup_django_model(AutoObsDC)
    assert carrier is not None and carrier.django_model is not None
    assert issubclass(carrier.django_model, DataclassTimescaleBase)
