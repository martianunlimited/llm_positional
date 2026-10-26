"""Pydantic v2 schemas for LLM Positional Encodings Lab."""

from pydantic import BaseModel, Field


class SinusoidalValue(BaseModel):
    """Represents a sinusoidal position embedding coordinate."""

    pos: int
    dim: int
    is_even: bool
    wavelength: float
    value: float


class RoPEPlaneRotation(BaseModel):
    """Rotary Position Embedding values in a 2D orthogonal plane."""

    pos: int
    subspace_idx: int
    theta_i: float
    angle_rad: float
    angle_deg: float
    cos_val: float
    sin_val: float


class RelativeDistancePoint(BaseModel):
    """Relative position distance and inner product decay factor."""

    distance: int
    theoretical_decay: float
    empirical_inner_product: float


class PermutationComparison(BaseModel):
    """Compares attention matrices with and without positional encoding."""

    sentence_original: str
    sentence_permuted: str
    tokens_original: list[str]
    tokens_permuted: list[str]
    pe_disabled_identical: bool
    attention_matrix_with_pe_1: list[list[float]]
    attention_matrix_with_pe_2: list[list[float]]


class PositionalPreset(BaseModel):
    """Educational scenario demonstrating a positional phenomenon."""

    id: str
    title: str
    category: str
    description: str
    explanation: str
    sentence_a: str
    sentence_b: str | None = None
    default_dim: int = 16
    default_base: float = 10000.0
    key_observations: list[str] = Field(default_factory=list)
