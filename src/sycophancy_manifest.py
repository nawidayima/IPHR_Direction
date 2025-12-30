"""Sycophancy Experiment Manifest: Deterministic Dataset Configuration.

This module provides a manifest-based configuration system for reproducible
sycophancy experiments. The manifest defines:

1. Which questions go in train vs eval splits
2. Which feedback templates to use
3. Version tracking for reproducibility

Manifest files are JSON and live in `experiments/{run_name}/manifest.json`.

Usage:
    from src.sycophancy_manifest import load_manifest, get_trajectory_specs

    manifest = load_manifest("experiments/canonical_v1/manifest.json")
    train_specs = get_trajectory_specs(manifest, split="train")
    eval_specs = get_trajectory_specs(manifest, split="eval")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import json

from .sycophancy import (
    FactualQuestion,
    QuestionCategory,
    SycophancyLabel,
    TrajectoryResult,
    CAPITAL_QUESTIONS,
    SCIENCE_QUESTIONS,
    GEOGRAPHY_QUESTIONS,
    STRONG_NEGATIVE_FEEDBACK_TEMPLATES,
    POSITIVE_FEEDBACK_TEMPLATES,
    NEGATIVE_FEEDBACK_TEMPLATES,
    SYSTEM_PROMPT,
    extract_answer,
    check_answer,
    label_trajectory,
)


# =============================================================================
# Question Bank (All 70 Questions in Fixed Order)
# =============================================================================

ALL_QUESTIONS: list[FactualQuestion] = (
    CAPITAL_QUESTIONS +      # 0-29: capitals
    SCIENCE_QUESTIONS +      # 30-49: science
    GEOGRAPHY_QUESTIONS      # 50-69: geography
)

# Feedback template banks (indexed for manifest references)
FEEDBACK_BANKS = {
    "STRONG_NEGATIVE": STRONG_NEGATIVE_FEEDBACK_TEMPLATES,
    "NEGATIVE": NEGATIVE_FEEDBACK_TEMPLATES,
    "POSITIVE": POSITIVE_FEEDBACK_TEMPLATES,
}


# =============================================================================
# Manifest Schema
# =============================================================================

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class ManifestSplit:
    """Configuration for a single data split (train or eval)."""
    question_indices: list[int]
    description: str = ""

    def __post_init__(self):
        # Validate indices
        for idx in self.question_indices:
            if not 0 <= idx < len(ALL_QUESTIONS):
                raise ValueError(f"Question index {idx} out of range [0, {len(ALL_QUESTIONS)})")


@dataclass
class ManifestFeedback:
    """Configuration for feedback templates."""
    template_bank: str = "STRONG_NEGATIVE"  # Key into FEEDBACK_BANKS
    use_all_templates: bool = True          # If True, use all templates; if False, use template_indices
    template_indices: list[int] = field(default_factory=list)  # Only used if use_all_templates=False

    @property
    def templates(self) -> list[str]:
        """Get the actual feedback templates based on configuration."""
        bank = FEEDBACK_BANKS.get(self.template_bank)
        if bank is None:
            raise ValueError(f"Unknown template bank: {self.template_bank}")

        if self.use_all_templates:
            return list(bank)
        else:
            return [bank[i] for i in self.template_indices]


@dataclass
class ExperimentManifest:
    """Complete manifest for a sycophancy experiment.

    This defines the exact dataset configuration for reproducibility.
    """
    schema_version: str
    name: str
    description: str
    created: str  # ISO format timestamp

    # Splits
    train: ManifestSplit
    eval: ManifestSplit

    # Feedback configuration
    feedback: ManifestFeedback

    # Trajectory ID format
    trajectory_id_format: str = "{split}_q{question_idx:03d}_f{feedback_idx}"

    # Optional: for legacy manifests that used random feedback
    legacy_feedback_mapping: dict[str, str] | None = None  # question_id -> feedback_text

    def validate(self) -> list[str]:
        """Validate the manifest and return list of errors (empty if valid)."""
        errors = []

        # Check schema version
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            errors.append(f"Schema version mismatch: {self.schema_version} != {MANIFEST_SCHEMA_VERSION}")

        # Check for overlapping indices
        train_set = set(self.train.question_indices)
        eval_set = set(self.eval.question_indices)
        overlap = train_set & eval_set
        if overlap:
            errors.append(f"Question indices appear in both train and eval: {overlap}")

        # Check feedback bank exists
        if self.feedback.template_bank not in FEEDBACK_BANKS:
            errors.append(f"Unknown feedback bank: {self.feedback.template_bank}")

        return errors

    @property
    def is_legacy(self) -> bool:
        """Check if this is a legacy manifest with random feedback."""
        return self.legacy_feedback_mapping is not None

    @property
    def n_train_trajectories(self) -> int:
        """Number of training trajectories."""
        if self.is_legacy:
            # Legacy: one trajectory per question (random feedback)
            return len(self.train.question_indices)
        n_feedbacks = len(self.feedback.templates)
        return len(self.train.question_indices) * n_feedbacks

    @property
    def n_eval_trajectories(self) -> int:
        """Number of evaluation trajectories."""
        if self.is_legacy:
            return len(self.eval.question_indices)
        n_feedbacks = len(self.feedback.templates)
        return len(self.eval.question_indices) * n_feedbacks

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "question_bank": {
                "source": "src/sycophancy.py",
                "total_questions": len(ALL_QUESTIONS),
            },
            "splits": {
                "train": {
                    "question_indices": self.train.question_indices,
                    "description": self.train.description,
                },
                "eval": {
                    "question_indices": self.eval.question_indices,
                    "description": self.eval.description,
                },
            },
            "feedback": {
                "template_bank": self.feedback.template_bank,
                "use_all_templates": self.feedback.use_all_templates,
                "template_indices": self.feedback.template_indices,
                "template_count": len(self.feedback.templates),
            },
            "trajectory_id_format": self.trajectory_id_format,
            "legacy_feedback_mapping": self.legacy_feedback_mapping,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentManifest":
        """Create manifest from dictionary."""
        splits = data.get("splits", {})
        feedback = data.get("feedback", {})

        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            description=data.get("description", ""),
            created=data.get("created", datetime.now().isoformat()),
            train=ManifestSplit(
                question_indices=splits.get("train", {}).get("question_indices", []),
                description=splits.get("train", {}).get("description", ""),
            ),
            eval=ManifestSplit(
                question_indices=splits.get("eval", {}).get("question_indices", []),
                description=splits.get("eval", {}).get("description", ""),
            ),
            feedback=ManifestFeedback(
                template_bank=feedback.get("template_bank", "STRONG_NEGATIVE"),
                use_all_templates=feedback.get("use_all_templates", True),
                template_indices=feedback.get("template_indices", []),
            ),
            trajectory_id_format=data.get("trajectory_id_format", "{split}_q{question_idx:03d}_f{feedback_idx}"),
            legacy_feedback_mapping=data.get("legacy_feedback_mapping"),
        )


# =============================================================================
# Trajectory Specification
# =============================================================================

@dataclass(frozen=True)
class TrajectorySpec:
    """Specification for a single trajectory (immutable, hashable).

    This defines WHAT to generate, not the result. The spec is deterministic
    and can be used to regenerate the exact same trajectory.
    """
    question_idx: int
    feedback_idx: int
    feedback_text: str  # Actual feedback string (for legacy compatibility)
    split: Literal["train", "eval"]
    trajectory_id: str

    @property
    def question(self) -> FactualQuestion:
        """Get the question for this trajectory."""
        return ALL_QUESTIONS[self.question_idx]

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "trajectory_id": self.trajectory_id,
            "question_idx": self.question_idx,
            "feedback_idx": self.feedback_idx,
            "feedback_text": self.feedback_text,
            "split": self.split,
            "question": self.question.question,
            "correct_answer": self.question.correct_answer,
            "category": self.question.category.value,
        }


def get_trajectory_specs(manifest: ExperimentManifest, split: Literal["train", "eval"]) -> list[TrajectorySpec]:
    """Generate all trajectory specs for a given split.

    Args:
        manifest: The experiment manifest
        split: "train" or "eval"

    Returns:
        List of TrajectorySpec objects in deterministic order
    """
    if split == "train":
        question_indices = manifest.train.question_indices
    elif split == "eval":
        question_indices = manifest.eval.question_indices
    else:
        raise ValueError(f"Unknown split: {split}")

    templates = manifest.feedback.templates
    specs = []

    for q_idx in question_indices:
        for f_idx, feedback_text in enumerate(templates):
            trajectory_id = manifest.trajectory_id_format.format(
                split=split,
                question_idx=q_idx,
                feedback_idx=f_idx,
            )
            specs.append(TrajectorySpec(
                question_idx=q_idx,
                feedback_idx=f_idx,
                feedback_text=feedback_text,
                split=split,
                trajectory_id=trajectory_id,
            ))

    return specs


# =============================================================================
# Manifest I/O
# =============================================================================

def load_manifest(path: Path | str) -> ExperimentManifest:
    """Load manifest from JSON file.

    Args:
        path: Path to manifest.json

    Returns:
        ExperimentManifest object
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    manifest = ExperimentManifest.from_dict(data)

    # Validate
    errors = manifest.validate()
    if errors:
        raise ValueError(f"Invalid manifest at {path}:\n" + "\n".join(f"  - {e}" for e in errors))

    return manifest


def save_manifest(manifest: ExperimentManifest, path: Path | str) -> None:
    """Save manifest to JSON file.

    Args:
        manifest: The manifest to save
        path: Path to save to (will create parent directories)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    print(f"Saved manifest to: {path}")


# =============================================================================
# Trajectory Generation
# =============================================================================

def generate_trajectory_from_spec(
    spec: TrajectorySpec,
    generate_fn: callable,
) -> TrajectoryResult:
    """Generate a trajectory from a spec using the provided generation function.

    Args:
        spec: The trajectory specification
        generate_fn: Function that takes messages list and returns response string
                    Signature: generate_fn(messages: list[dict]) -> str

    Returns:
        TrajectoryResult with all fields populated
    """
    question = spec.question
    feedback = spec.feedback_text

    # Turn 1: Ask question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.question},
    ]

    first_response = generate_fn(messages)
    first_answer = extract_answer(first_response, question.category)
    first_correct = check_answer(first_answer, question)

    # Turn 2: Give feedback, get second response
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": feedback})

    second_response = generate_fn(messages)
    second_answer = extract_answer(second_response, question.category)

    # Determine if answer changed
    if first_answer and second_answer:
        answer_changed = first_answer.lower().strip() != second_answer.lower().strip()
    else:
        answer_changed = False

    # Label the trajectory
    label = label_trajectory(first_answer, second_answer, question, "negative")

    return TrajectoryResult(
        question_id=spec.trajectory_id,
        question=question.question,
        correct_answer=question.correct_answer,
        category=question.category.value,
        first_response=first_response,
        first_answer=first_answer,
        first_correct=first_correct,
        feedback_type="negative",
        feedback=feedback,
        second_response=second_response,
        second_answer=second_answer,
        answer_changed=answer_changed,
        label=label,
    )


# =============================================================================
# Manifest Factories
# =============================================================================

def create_canonical_v1_manifest() -> ExperimentManifest:
    """Create the canonical v1 manifest with proper train/eval split.

    Train: Questions 0-49 (50 questions) - for probe training
    Eval: Questions 50-69 (20 questions) - held-out for steering evaluation
    All 8 strong negative feedback templates per question
    """
    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="canonical_v1",
        description="Canonical sycophancy dataset with held-out eval. Train on questions 0-49, evaluate steering on 50-69.",
        created=datetime.now().isoformat(),
        train=ManifestSplit(
            question_indices=list(range(50)),
            description="First 50 questions (30 capitals, 16 science, 4 geography) for probe training",
        ),
        eval=ManifestSplit(
            question_indices=list(range(50, 70)),
            description="Last 20 questions (4 science, 16 geography) held-out for steering evaluation",
        ),
        feedback=ManifestFeedback(
            template_bank="STRONG_NEGATIVE",
            use_all_templates=True,
        ),
    )


def create_legacy_v0_manifest(
    trajectories_csv_path: Path | str,
) -> ExperimentManifest:
    """Create a legacy v0 manifest from existing proven checkpoint.

    This analyzes the existing trajectories CSV to extract which questions
    and feedback messages were used, preserving the exact configuration
    that achieved AUC=0.933.

    Args:
        trajectories_csv_path: Path to the proven sycophancy.csv

    Returns:
        ExperimentManifest documenting the legacy configuration
    """
    import pandas as pd

    df = pd.read_csv(trajectories_csv_path)

    # Extract unique questions used (by matching to ALL_QUESTIONS)
    question_to_idx = {q.question: i for i, q in enumerate(ALL_QUESTIONS)}

    used_indices = set()
    feedback_mapping = {}

    for _, row in df.iterrows():
        q_text = row["question"]
        if q_text in question_to_idx:
            used_indices.add(question_to_idx[q_text])

        # Store the exact feedback used for each question_id
        qid = row.get("question_id", "")
        feedback = row.get("feedback", "")
        if qid and feedback:
            feedback_mapping[qid] = feedback

    used_indices = sorted(used_indices)

    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="legacy_v0",
        description="Legacy manifest documenting the proven checkpoint that achieved AUC=0.933. Uses random feedback selection (not deterministic).",
        created=datetime.now().isoformat(),
        train=ManifestSplit(
            question_indices=used_indices,
            description=f"All {len(used_indices)} questions used in original experiment (random feedback per question)",
        ),
        eval=ManifestSplit(
            question_indices=[],  # Legacy had no held-out eval
            description="No held-out eval in legacy experiment",
        ),
        feedback=ManifestFeedback(
            template_bank="STRONG_NEGATIVE",
            use_all_templates=False,  # Legacy used random single feedback per question
            template_indices=[],  # Not applicable - random selection
        ),
        legacy_feedback_mapping=feedback_mapping,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def print_manifest_summary(manifest: ExperimentManifest) -> None:
    """Print a human-readable summary of a manifest."""
    print("=" * 70)
    print(f"MANIFEST: {manifest.name}")
    if manifest.is_legacy:
        print("  [LEGACY - uses random feedback, not fully reproducible]")
    print("=" * 70)
    print(f"Description: {manifest.description}")
    print(f"Created: {manifest.created}")
    print(f"Schema version: {manifest.schema_version}")
    print()

    print("Splits:")
    print(f"  Train: {len(manifest.train.question_indices)} questions")
    if manifest.train.question_indices:
        indices = manifest.train.question_indices
        if len(indices) > 8:
            print(f"    Indices: {indices[:5]}...{indices[-3:]}")
        else:
            print(f"    Indices: {indices}")
    print(f"    {manifest.train.description}")
    print(f"  Eval: {len(manifest.eval.question_indices)} questions")
    if manifest.eval.question_indices:
        indices = manifest.eval.question_indices
        if len(indices) > 8:
            print(f"    Indices: {indices[:5]}...{indices[-3:]}")
        else:
            print(f"    Indices: {indices}")
    print(f"    {manifest.eval.description}")
    print()

    print("Feedback:")
    print(f"  Bank: {manifest.feedback.template_bank}")
    if manifest.is_legacy:
        print(f"  Mode: Random single feedback per question (legacy)")
        print(f"  Feedback mappings stored: {len(manifest.legacy_feedback_mapping or {})}")
    else:
        print(f"  Use all templates: {manifest.feedback.use_all_templates}")
        print(f"  Template count: {len(manifest.feedback.templates)}")
    print()

    print("Trajectory counts:")
    print(f"  Train: {manifest.n_train_trajectories}")
    print(f"  Eval: {manifest.n_eval_trajectories}")
    print(f"  Total: {manifest.n_train_trajectories + manifest.n_eval_trajectories}")
    print("=" * 70)


if __name__ == "__main__":
    # Demo: create and print canonical v1 manifest
    manifest = create_canonical_v1_manifest()
    print_manifest_summary(manifest)
