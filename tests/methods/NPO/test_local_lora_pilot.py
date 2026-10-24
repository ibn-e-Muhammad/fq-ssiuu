import pytest

from experiments.reproductions.NPO.local_lora_pilot import validate_settings


@pytest.mark.parametrize(
    "settings",
    [
        (1, 3, 1, 1, 1),
        (62, 500, 8, 16, 32),
    ],
)
def test_local_pilot_accepts_bounded_settings(settings):
    validate_settings(*settings)


@pytest.mark.parametrize(
    "settings",
    [
        (0, 500, 4, 8, 16),
        (63, 500, 4, 8, 16),
        (1, 501, 4, 8, 16),
        (1, 500, 9, 8, 16),
        (1, 500, 4, 17, 16),
        (1, 500, 4, 8, 33),
    ],
)
def test_local_pilot_rejects_unbounded_settings(settings):
    with pytest.raises(ValueError):
        validate_settings(*settings)
