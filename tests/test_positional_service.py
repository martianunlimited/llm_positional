"""Unit tests for PositionalService and core mathematical formulas."""

import math

from src.core.positional_service import (
    PositionalService,
    apply_rope_2d,
    compute_rope_relative_inner_product,
    compute_sinusoidal_pe,
)


def test_sinusoidal_boundary_conditions() -> None:
    pe = compute_sinusoidal_pe(max_pos=16, dim=8)
    assert len(pe) == 16
    assert len(pe[0]) == 8

    # At position 0:
    # Even dimensions: sin(0) = 0.0
    for i in range(0, 8, 2):
        assert math.isclose(pe[0][i], 0.0, abs_tol=1e-5)

    # Odd dimensions: cos(0) = 1.0
    for i in range(1, 8, 2):
        assert math.isclose(pe[0][i], 1.0, abs_tol=1e-5)

    # Values must remain bounded in [-1.0, 1.0]
    for row in pe:
        for val in row:
            assert -1.0 <= val <= 1.0


def test_sinusoidal_wavelength_progression() -> None:
    dim = 16
    base = 10000.0
    pe = compute_sinusoidal_pe(max_pos=32, dim=dim, base=base)

    # Dimension 0 has highest frequency / shortest wavelength: 2 * pi * 1.0
    # Higher dimensions have lower frequency / longer wavelength: 2 * pi * (10000 ^ (2i/dim))
    # Test that dim 0 alternates faster across positions than dim 14
    sign_changes_dim0 = sum(1 for p in range(1, 32) if (pe[p][0] >= 0) != (pe[p - 1][0] >= 0))
    sign_changes_dim14 = sum(1 for p in range(1, 32) if (pe[p][14] >= 0) != (pe[p - 1][14] >= 0))
    assert sign_changes_dim0 > sign_changes_dim14


def test_rope_norm_preservation() -> None:
    # RoPE rotates vectors on a 2D plane: length must be preserved exactly!
    vec = (3.0, 4.0)
    orig_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)  # 5.0

    theta = 0.45
    for pos in [1, 5, 20, 100]:
        rotated = apply_rope_2d(vec, pos=pos, theta_i=theta)
        rot_norm = math.sqrt(rotated[0] ** 2 + rotated[1] ** 2)
        assert math.isclose(orig_norm, rot_norm, rel_tol=1e-5)


def test_rope_relative_shift_property() -> None:
    # Inner product between rotated Q at pos m and rotated K at pos n
    # depends only on the relative distance (m - n)!
    theta = 0.35
    q = (1.0, 0.5)
    k = (0.8, 1.2)

    # Pair 1: pos 10 and pos 4 (distance = 6)
    q_10 = apply_rope_2d(q, pos=10, theta_i=theta)
    k_4 = apply_rope_2d(k, pos=4, theta_i=theta)
    dot_1 = q_10[0] * k_4[0] + q_10[1] * k_4[1]

    # Pair 2: pos 26 and pos 20 (distance = 6)
    q_26 = apply_rope_2d(q, pos=26, theta_i=theta)
    k_20 = apply_rope_2d(k, pos=20, theta_i=theta)
    dot_2 = q_26[0] * k_20[0] + q_26[1] * k_20[1]

    assert math.isclose(dot_1, dot_2, rel_tol=1e-5)

    # Theoretical relative formula
    rel_dot = compute_rope_relative_inner_product(10, 4, theta)
    assert math.isclose(
        rel_dot,
        math.cos(6 * theta),
        rel_tol=1e-5,
    )


def test_rope_base_frequency_scaling_llama3() -> None:
    # Standard LLaMA 1/2 base = 10,000; LLaMA 3 base = 500,000
    # Higher base slows down rotation, allowing long context extension
    service = PositionalService()
    subspaces_std = service.generate_rope_subspaces(pos=10, dim=16, base=10000.0)
    subspaces_llama3 = service.generate_rope_subspaces(pos=10, dim=16, base=500000.0)

    # Higher base should produce smaller rotation angles for the same position
    assert subspaces_llama3[2].angle_rad < subspaces_std[2].angle_rad


def test_permutation_invariance_demonstration() -> None:
    service = PositionalService()
    res = service.compare_permutation("Dog bites man", "Man bites dog")

    # Without PE, attention weights between identical word tokens are symmetric
    assert res.pe_disabled_identical

    # With PE, the attention matrix for "Dog bites man" is distinct from "Man bites dog"
    assert res.attention_matrix_with_pe_1 != res.attention_matrix_with_pe_2
