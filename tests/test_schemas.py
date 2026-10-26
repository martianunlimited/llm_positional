"""Unit tests for Positional Encodings Pydantic schemas."""

from src.schemas import (
    PermutationComparison,
    PositionalPreset,
    RelativeDistancePoint,
    RoPEPlaneRotation,
    SinusoidalValue,
)


def test_sinusoidal_value_schema() -> None:
    val = SinusoidalValue(
        pos=0,
        dim=0,
        is_even=True,
        wavelength=6.283,
        value=0.0,
    )
    assert val.pos == 0
    assert val.dim == 0
    assert val.is_even
    assert val.value == 0.0


def test_rope_plane_rotation_schema() -> None:
    rot = RoPEPlaneRotation(
        pos=1,
        subspace_idx=0,
        theta_i=1.0,
        angle_rad=1.0,
        angle_deg=57.295,
        cos_val=0.5403,
        sin_val=0.8415,
    )
    assert rot.pos == 1
    assert rot.subspace_idx == 0
    assert rot.cos_val == 0.5403


def test_relative_distance_point_schema() -> None:
    pt = RelativeDistancePoint(
        distance=5,
        theoretical_decay=0.88,
        empirical_inner_product=0.875,
    )
    assert pt.distance == 5
    assert pt.theoretical_decay == 0.88


def test_permutation_comparison_schema() -> None:
    comp = PermutationComparison(
        sentence_original="Dog bites man",
        sentence_permuted="Man bites dog",
        tokens_original=["Dog", "bites", "man"],
        tokens_permuted=["Man", "bites", "dog"],
        pe_disabled_identical=True,
        attention_matrix_with_pe_1=[[1.0, 0.0, 0.0]],
        attention_matrix_with_pe_2=[[0.8, 0.2, 0.0]],
    )
    assert comp.pe_disabled_identical
    assert len(comp.tokens_original) == 3


def test_positional_preset_schema() -> None:
    preset = PositionalPreset(
        id="dog-man-permutation",
        title="Bag-of-Words Dilemma: Dog Bites Man",
        category="Permutation",
        description="Attention symmetry without positional encoding",
        explanation="Without positional information, both sentences generate identical embeddings.",
        sentence_a="Dog bites man",
        sentence_b="Man bites dog",
        default_dim=16,
        default_base=10000.0,
        key_observations=["Syntax is completely lost without position encoding"],
    )
    assert preset.id == "dog-man-permutation"
    assert preset.sentence_b == "Man bites dog"
