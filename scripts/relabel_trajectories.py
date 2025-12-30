#!/usr/bin/env python3
"""Relabel trajectory CSVs with updated extraction logic.

This script re-processes existing trajectory CSVs using the fixed
extract_answer() and label_trajectory() functions to correct labeling errors.

Usage:
    python scripts/relabel_trajectories.py [input_csv] [output_csv]

    If no arguments provided, processes the default eval_trajectories.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.sycophancy import (
    QuestionCategory,
    FactualQuestion,
    extract_answer,
    check_answer,
    label_trajectory,
    CAPITAL_QUESTIONS,
    SCIENCE_QUESTIONS,
    GEOGRAPHY_QUESTIONS,
)

# Build question lookup
ALL_QUESTIONS = CAPITAL_QUESTIONS + SCIENCE_QUESTIONS + GEOGRAPHY_QUESTIONS
QUESTION_LOOKUP = {q.question: q for q in ALL_QUESTIONS}


def relabel_row(row: pd.Series) -> dict:
    """Relabel a single trajectory row with fixed extraction logic."""
    # Get the question object
    question_text = row['question']
    if question_text not in QUESTION_LOOKUP:
        # Try to find by correct answer
        for q in ALL_QUESTIONS:
            if q.correct_answer == row['correct_answer']:
                question = q
                break
        else:
            print(f"Warning: Could not find question: {question_text[:50]}...")
            return row.to_dict()
    else:
        question = QUESTION_LOOKUP[question_text]

    # Re-extract answers
    new_second_answer = extract_answer(row['second_response'], question.category)

    # Check correctness
    new_second_correct = check_answer(new_second_answer, question)

    # Determine answer_changed
    first_answer = row['first_answer']
    if first_answer and new_second_answer:
        new_answer_changed = first_answer.lower().strip() != new_second_answer.lower().strip()
    else:
        new_answer_changed = False

    # Re-label
    new_label = label_trajectory(
        first_answer,
        new_second_answer,
        question,
        row['feedback_type']
    )

    # Return updated row
    result = row.to_dict()
    result['second_answer'] = new_second_answer
    result['answer_changed'] = new_answer_changed
    result['label'] = new_label.value

    return result


def relabel_csv(input_path: Path, output_path: Path):
    """Relabel all trajectories in a CSV file."""
    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Processing {len(df)} trajectories...")

    # Track changes
    original_labels = df['label'].value_counts().to_dict()

    # Process each row
    new_rows = []
    for idx, row in df.iterrows():
        new_row = relabel_row(row)
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    # Report changes
    new_labels = new_df['label'].value_counts().to_dict()

    print("\nLabel distribution changes:")
    print(f"  Original: {original_labels}")
    print(f"  New:      {new_labels}")

    # Count specific changes
    changes = (df['label'] != new_df['label']).sum()
    print(f"\nTotal rows changed: {changes}")

    # Show some examples of changes
    changed_mask = df['label'] != new_df['label']
    if changed_mask.any():
        print("\nExample changes:")
        for idx in df[changed_mask].head(5).index:
            print(f"  {df.loc[idx, 'question_id']}: {df.loc[idx, 'label']} -> {new_df.loc[idx, 'label']}")
            print(f"    Old answer: {df.loc[idx, 'second_answer']}")
            print(f"    New answer: {new_df.loc[idx, 'second_answer']}")

    # Save
    new_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


def main():
    if len(sys.argv) >= 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        input_path = Path(sys.argv[1])
        output_path = input_path.with_suffix('.relabeled.csv')
    else:
        # Default paths
        base_dir = Path(__file__).parent.parent
        input_path = base_dir / "experiments/canonical_v1/trajectories/eval_trajectories.csv"
        output_path = base_dir / "experiments/canonical_v1/trajectories/eval_trajectories_relabeled.csv"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    relabel_csv(input_path, output_path)


if __name__ == "__main__":
    main()