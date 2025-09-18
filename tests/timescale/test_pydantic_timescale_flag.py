from pydantic import BaseModel

from pydantic2django.pydantic.generator import StaticPydanticModelGenerator


class ObviousHypertable(BaseModel):
    timestamp: int
    values: list[float]


def test_pydantic_generator_timescale_can_be_disabled(tmp_path):
    # With enable_timescale=False, a model that would normally be hypertable should not use Timescale base
    gen = StaticPydanticModelGenerator(
        output_path=str(tmp_path / "out.py"), packages=["tests"], app_label="tests", enable_timescale=False
    )
    carrier = gen.setup_django_model(ObviousHypertable)
    assert carrier is not None and carrier.django_model is not None

    # The selected base should be the configured base_model_class, not a Timescale variant
    base_name = carrier.django_model.__bases__[0].__name__ if carrier.django_model.__bases__ else ""
    assert not base_name.endswith("TimescaleBase")
