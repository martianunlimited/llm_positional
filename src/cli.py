"""Typer CLI interface for LLM Positional Encodings Lab."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.positional_service import PositionalService
from src.core.preset_data import get_all_presets, get_preset_by_id

app = typer.Typer(
    name="positional-cli",
    help="Interactive CLI tool for exploring Sinusoidal and Rotary (RoPE) Positional Encodings.",
)
console = Console()


@app.command()
def permutation(
    sentence_a: str = typer.Option("Dog bites man", "--sent-a", "-a"),
    sentence_b: str = typer.Option("Man bites dog", "--sent-b", "-b"),
) -> None:
    """Demonstrate why Transformers are permutation-invariant without positional encodings."""
    service = PositionalService()
    res = service.compare_permutation(sentence_a, sentence_b)

    console.print(
        Panel.fit(
            f"[bold cyan]Sentence A:[/bold cyan] {sentence_a}\n"
            f"[bold magenta]Sentence B:[/bold magenta] {sentence_b}\n\n"
            f"[bold yellow]Without Positional Encoding:[/bold yellow]\n"
            f"Attention scores between identical words are mathematically symmetric!\n"
            f"The model CANNOT determine who bit whom (Permutation Invariant / Bag-of-Words).\n\n"
            f"[bold green]With Positional Encoding:[/bold green]\n"
            f"Coordinates shift according to token index, breaking symmetry and encoding syntax.",
            title="🐕 Bag-of-Words Permutation Dilemma",
            border_style="yellow",
        )
    )

    table_a = Table(title=f"Attention Matrix (With PE): '{sentence_a}'", show_header=True)
    table_a.add_column("Token", style="cyan")
    for t in res.tokens_original:
        table_a.add_column(t, justify="center")

    for i, tok in enumerate(res.tokens_original):
        row = [f"{i}: {tok}"]
        for j in range(len(res.tokens_original)):
            val = res.attention_matrix_with_pe_1[i][j]
            row.append(f"{val:.3f}")
        table_a.add_row(*row)

    console.print(table_a)


@app.command()
def sinusoidal(
    max_pos: int = typer.Option(8, "--pos", "-p", help="Number of positions to show"),
    dim: int = typer.Option(8, "--dim", "-d", help="Embedding dimension"),
) -> None:
    """Display the Vaswani Sinusoidal Positional Encoding matrix."""
    service = PositionalService()
    matrix = service.generate_sinusoidal_matrix(max_pos=max_pos, dim=dim)

    table = Table(title=f"Vaswani Sinusoidal PE Matrix ({max_pos} pos x {dim} dim)")
    table.add_column("Pos", style="cyan", justify="right")

    for d in range(dim):
        name = f"d{d} (sin)" if d % 2 == 0 else f"d{d} (cos)"
        table.add_column(name, justify="right")

    for pos in range(max_pos):
        row = [str(pos)]
        for d in range(dim):
            val = matrix[pos][d]
            row.append(f"{val:+.3f}")
        table.add_row(*row)

    console.print(table)


@app.command()
def rope(
    pos: int = typer.Option(1, "--pos", "-p", help="Token position index m"),
    dim: int = typer.Option(8, "--dim", "-d", help="Embedding dimension"),
    base: float = typer.Option(10000.0, "--base", "-b", help="RoPE base frequency theta"),
) -> None:
    """Inspect RoPE 2D orthogonal plane rotation angles."""
    service = PositionalService()
    subspaces = service.generate_rope_subspaces(pos=pos, dim=dim, base=base)

    console.print(
        Panel.fit(
            f"[bold cyan]Token Position (m):[/bold cyan] {pos}\n"
            f"[bold]Dimension:[/bold] {dim} ({dim // 2} 2D Orthogonal Planes)\n"
            f"[bold]Frequency Base (theta):[/bold] {base:,.0f}",
            title="🔄 Modern RoPE (Rotary Position Embeddings)",
            border_style="cyan",
        )
    )

    table = Table(title="2D Subspace Rotation Angles", show_header=True)
    table.add_column("Plane", justify="center", style="dim")
    table.add_column("Dimensions", justify="center")
    table.add_column("Theta_i", justify="right")
    table.add_column("Angle (rad)", justify="right")
    table.add_column("Angle (deg)", justify="right", style="bold green")
    table.add_column("cos(m*theta)", justify="right")
    table.add_column("sin(m*theta)", justify="right")

    for s in subspaces:
        dims_str = f"d{s.subspace_idx * 2}, d{s.subspace_idx * 2 + 1}"
        table.add_row(
            f"Plane {s.subspace_idx}",
            dims_str,
            f"{s.theta_i:.5f}",
            f"{s.angle_rad:.4f}",
            f"{s.angle_deg:.1f}°",
            f"{s.cos_val:+.4f}",
            f"{s.sin_val:+.4f}",
        )

    console.print(table)


@app.command()
def presets() -> None:
    """List all available educational positional presets."""
    all_presets = get_all_presets()
    table = Table(title="📚 Educational Positional Presets", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("Category", style="magenta")

    for p in all_presets:
        table.add_row(p.id, p.title, p.category)

    console.print(table)


@app.command()
def inspect(preset_id: str = typer.Argument(..., help="Preset ID to inspect")) -> None:
    """Inspect a preset scenario with in-depth positional explanation."""
    preset = get_preset_by_id(preset_id)
    if not preset:
        console.print(f"[red]Error: Preset '{preset_id}' not found.[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Title:[/bold cyan] {preset.title}\n"
            f"[bold magenta]Category:[/bold magenta] {preset.category}\n\n"
            f"[bold green]Explanation:[/bold green]\n{preset.explanation}",
            title=f"Scenario: {preset.id}",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
