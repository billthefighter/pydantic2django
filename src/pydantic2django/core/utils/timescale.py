from __future__ import annotations

from collections.abc import Iterable


class TimeseriesTimestampMissingError(Exception):
    """Raised when a Timescale-enabled model requires 'time' but no source timestamp is found."""

    def __init__(self, model_name: str, *, attempted_aliases: list[str]) -> None:
        message = (
            f"Timescale model '{model_name}' requires a non-null 'time' value, "
            f"but none of the expected timestamp attributes were found: {attempted_aliases}. "
            f"Provide one of these attributes in the source payload or customize mapping."
        )
        super().__init__(message)


# Canonical list of accepted alias names (already normalized to snake_case)
TIMESERIES_TIME_ALIASES: tuple[str, ...] = (
    "creation_time",
    "timestamp",
    "time_stamp",
    "effective_time",
    "date_time",
    "datetime",
)


def has_timescale_time_field(model_field_names: set[str]) -> bool:
    """Return True if the model declares a 'time' field."""
    return "time" in model_field_names


def map_time_alias_into_time(
    field_values: dict, *, aliases: Iterable[str] = TIMESERIES_TIME_ALIASES
) -> tuple[bool, list[str]]:
    """
    Map the first present alias from 'aliases' into the canonical 'time' key.

    Returns (mapped, attempted_aliases)
    """
    attempted = list(aliases)
    if "time" in field_values:
        return True, attempted
    for name in attempted:
        if name in field_values:
            field_values["time"] = field_values.pop(name)
            return True, attempted
    return False, attempted


def ensure_time_or_raise(
    model_name: str,
    field_values: dict,
    model_field_names: set[str],
    *,
    aliases: Iterable[str] = TIMESERIES_TIME_ALIASES,
) -> None:
    """
    If the model declares a 'time' field, ensure it is present in field_values,
    attempting to remap aliases first; raise a clear error if still missing.
    """
    if not has_timescale_time_field(model_field_names):
        return
    mapped, attempted = map_time_alias_into_time(field_values, aliases=aliases)
    if not mapped:
        raise TimeseriesTimestampMissingError(model_name, attempted_aliases=list(attempted))
