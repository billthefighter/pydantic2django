from pydantic import BaseModel

from pydantic2django.django.timescale.bases import PydanticTimescaleBase
from pydantic2django.pydantic.generator import StaticPydanticModelGenerator


class AutoObs(BaseModel):
    device_id: str
    timestamp: int
    values: list[float]


def test_pydantic_generator_picks_timescale_base_automatically():
    gen = StaticPydanticModelGenerator(output_path="/tmp/out.py", packages=["tests"], app_label="tests")
    carrier = gen.setup_django_model(AutoObs)
    assert carrier is not None and carrier.django_model is not None
    assert issubclass(carrier.django_model, PydanticTimescaleBase)
