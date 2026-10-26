"""Unit tests for Positional Encodings preset scenarios."""

from src.core.preset_data import get_all_presets, get_preset_by_id


def test_presets_exist() -> None:
    presets = get_all_presets()
    assert len(presets) >= 4


def test_presets_validation() -> None:
    presets = get_all_presets()
    categories = {p.category for p in presets}
    assert "Permutation Dilemma" in categories
    assert "Sinusoidal Waves" in categories
    assert "Rotary Embedding (RoPE)" in categories
    assert "Long-Context Extrapolation" in categories

    for p in presets:
        assert p.id
        assert p.title
        assert len(p.explanation) > 30
        assert len(p.key_observations) >= 1


def test_get_preset_by_id() -> None:
    preset = get_preset_by_id("dog-man-permutation")
    assert preset is not None
    assert preset.sentence_b is not None

    missing = get_preset_by_id("non-existent")
    assert missing is None
