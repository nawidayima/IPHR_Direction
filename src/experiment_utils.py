"""Experiment tracking utilities for reproducibility and organization.

Each experiment run is self-contained with:
- config.json: Hyperparameters, model info, layer selection
- results.json: Metrics, contradiction rates, accuracy scores
- trajectories/: Question pairs and model responses (CSVs)
- activations/: Cached activation tensors
- plots/: Visualizations
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment metadata
    name: str
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda"

    # Data configuration
    domains: list[str] = field(default_factory=lambda: ["geography", "dates", "population"])
    max_pairs_per_domain: int = 50

    # Generation configuration
    max_new_tokens: int = 300
    temperature: float = 0.0  # Greedy decoding
    do_sample: bool = False

    # Probing configuration (for Phase 2)
    layers_to_probe: list[int] = field(default_factory=lambda: [8, 16, 24, 31])
    token_position: str = "last"  # "last" or "decision"

    # Additional settings
    random_seed: int = 42
    notes: str = ""


@dataclass
class ExperimentResults:
    """Results from an experiment run."""

    # Overall metrics
    total_pairs: int = 0
    total_contradictions: int = 0
    contradiction_rate: float = 0.0

    # Per-domain metrics
    domain_metrics: dict[str, dict] = field(default_factory=dict)

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Probing results (Phase 2)
    probe_accuracy_by_layer: dict[int, float] = field(default_factory=dict)

    # Error tracking
    errors: list[str] = field(default_factory=list)


def get_experiments_dir() -> Path:
    """Get the experiments directory path."""
    # Look for project root by finding pyproject.toml
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current / "experiments"
        current = current.parent

    # Fallback to current directory
    return Path.cwd() / "experiments"


def create_experiment_run(
    name: str,
    config: ExperimentConfig | None = None,
) -> Path:
    """Create a new experiment run folder.

    Args:
        name: Short name for the experiment (e.g., "baseline", "larger_dataset")
        config: Experiment configuration. Created with defaults if not provided.

    Returns:
        Path to the experiment run folder.
    """
    experiments_dir = get_experiments_dir()
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Create unique run folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{name}"
    run_dir = experiments_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "trajectories").mkdir(exist_ok=True)
    (run_dir / "activations").mkdir(exist_ok=True)

    # Create config if not provided
    if config is None:
        config = ExperimentConfig(name=name)

    # Save config
    save_config(run_dir, config)

    return run_dir


def save_config(run_dir: Path, config: ExperimentConfig) -> None:
    """Save experiment configuration to config.json."""
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def load_config(run_dir: Path) -> ExperimentConfig:
    """Load experiment configuration from config.json."""
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        data = json.load(f)
    return ExperimentConfig(**data)


def save_results(run_dir: Path, results: ExperimentResults) -> None:
    """Save experiment results to results.json."""
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(asdict(results), f, indent=2)


def load_results(run_dir: Path) -> ExperimentResults:
    """Load experiment results from results.json."""
    results_path = run_dir / "results.json"
    with open(results_path) as f:
        data = json.load(f)
    return ExperimentResults(**data)


def log_domain_metrics(
    results: ExperimentResults,
    domain: str,
    total_pairs: int,
    contradictions: int,
    correct_a: int,
    correct_b: int,
) -> None:
    """Log metrics for a specific domain."""
    results.domain_metrics[domain] = {
        "total_pairs": total_pairs,
        "contradictions": contradictions,
        "contradiction_rate": contradictions / total_pairs if total_pairs > 0 else 0.0,
        "correct_a": correct_a,
        "correct_b": correct_b,
        "accuracy_a": correct_a / total_pairs if total_pairs > 0 else 0.0,
        "accuracy_b": correct_b / total_pairs if total_pairs > 0 else 0.0,
    }


def finalize_results(results: ExperimentResults) -> None:
    """Calculate overall metrics from domain metrics."""
    total_pairs = sum(m["total_pairs"] for m in results.domain_metrics.values())
    total_contradictions = sum(m["contradictions"] for m in results.domain_metrics.values())

    results.total_pairs = total_pairs
    results.total_contradictions = total_contradictions
    results.contradiction_rate = total_contradictions / total_pairs if total_pairs > 0 else 0.0


def list_experiments() -> list[Path]:
    """List all experiment run folders."""
    experiments_dir = get_experiments_dir()
    if not experiments_dir.exists():
        return []

    runs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,  # Most recent first
    )
    return runs


def print_experiment_summary(run_dir: Path) -> None:
    """Print a summary of an experiment run."""
    config = load_config(run_dir)
    results_path = run_dir / "results.json"

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"Timestamp: {config.timestamp}")
    print(f"{'='*60}")

    if results_path.exists():
        results = load_results(run_dir)
        print(f"\nTotal pairs: {results.total_pairs}")
        print(f"Contradictions: {results.total_contradictions}")
        print(f"Contradiction rate: {results.contradiction_rate:.1%}")

        if results.domain_metrics:
            print("\nPer-domain breakdown:")
            for domain, metrics in results.domain_metrics.items():
                print(f"  {domain}: {metrics['contradictions']}/{metrics['total_pairs']} "
                      f"({metrics['contradiction_rate']:.1%})")
    else:
        print("\nNo results yet.")

    print(f"\nFolder: {run_dir}")
