"""Expand sycophancy dataset with strong negative feedback.

This script generates additional sycophancy trajectories focusing on:
1. Science and geography questions (higher sycophancy rate)
2. Strong negative feedback (more likely to induce sycophancy)

Run on Colab with GPU for faster generation.

Usage:
    python scripts/expand_sycophancy_data.py --output experiments/run_XXXX/trajectories/sycophancy_expanded.csv
"""

import argparse
import random
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Will be imported after package installation
# from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Expand sycophancy dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/sycophancy_expanded.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--existing",
        type=str,
        default=None,
        help="Path to existing sycophancy.csv to append to",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of runs per question with strong feedback",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["science", "geography"],
        help="Categories to focus on (default: science, geography)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Imports (delayed to allow script to show help without GPU)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.sycophancy import (
        QuestionCategory,
        SycophancyLabel,
        TrajectoryResult,
        SYSTEM_PROMPT,
        SCIENCE_QUESTIONS,
        GEOGRAPHY_QUESTIONS,
        STRONG_NEGATIVE_FEEDBACK_TEMPLATES,
        extract_answer,
        check_answer,
        label_trajectory,
    )

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    random.seed(42)
    torch.manual_seed(42)

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get questions
    questions = []
    if "science" in args.categories:
        questions.extend(SCIENCE_QUESTIONS)
    if "geography" in args.categories:
        questions.extend(GEOGRAPHY_QUESTIONS)

    print(f"Using {len(questions)} questions from: {args.categories}")

    # Generation function
    def generate_response(messages, max_new_tokens=100):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return response.strip()

    # Generate trajectories
    all_trajectories = []

    for run_idx in range(args.n_runs):
        print(f"\nRun {run_idx + 1}/{args.n_runs}")

        for q_idx, q in enumerate(tqdm(questions, desc="Generating")):
            question_id = f"q_strong_{run_idx}_{q_idx:03d}"

            # Pick a random strong feedback
            feedback = random.choice(STRONG_NEGATIVE_FEEDBACK_TEMPLATES)

            try:
                # Turn 1: Ask question
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q.question},
                ]
                first_response = generate_response(messages)
                first_answer = extract_answer(first_response, q.category)
                first_correct = check_answer(first_answer, q)

                # Turn 2: Strong negative feedback
                messages.append({"role": "assistant", "content": first_response})
                messages.append({"role": "user", "content": feedback})
                second_response = generate_response(messages)
                second_answer = extract_answer(second_response, q.category)

                # Check if answer changed
                if first_answer and second_answer:
                    answer_changed = first_answer.lower().strip() != second_answer.lower().strip()
                else:
                    answer_changed = False

                # Label
                label = label_trajectory(first_answer, second_answer, q, "negative")

                traj = TrajectoryResult(
                    question_id=question_id,
                    question=q.question,
                    correct_answer=q.correct_answer,
                    category=q.category.value,
                    first_response=first_response,
                    first_answer=first_answer,
                    first_correct=first_correct,
                    feedback_type="negative_strong",
                    feedback=feedback,
                    second_response=second_response,
                    second_answer=second_answer,
                    answer_changed=answer_changed,
                    label=label,
                )
                all_trajectories.append(traj)

            except Exception as e:
                print(f"\nError at {question_id}: {e}")

            # Clear cache periodically
            if q_idx % 10 == 0:
                torch.cuda.empty_cache()

    # Convert to DataFrame
    df = pd.DataFrame([t.to_dict() for t in all_trajectories])

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    valid_neg = df[df["first_correct"] == True]
    n_syc = (valid_neg["label"] == "sycophantic").sum()
    n_maintained = (valid_neg["label"] == "maintained").sum()
    n_total = len(valid_neg)

    print(f"Total trajectories: {len(df)}")
    print(f"Valid (first correct): {n_total}")
    print(f"Sycophantic: {n_syc} ({n_syc/n_total*100:.1f}%)")
    print(f"Maintained: {n_maintained} ({n_maintained/n_total*100:.1f}%)")

    # Optionally merge with existing
    if args.existing and Path(args.existing).exists():
        existing_df = pd.read_csv(args.existing)
        print(f"\nMerging with existing data ({len(existing_df)} rows)")
        df = pd.concat([existing_df, df], ignore_index=True)

        # Recount
        valid_all = df[(df["first_correct"] == True) & (df["feedback_type"].str.contains("negative"))]
        n_syc_total = (valid_all["label"] == "sycophantic").sum()
        n_maintained_total = (valid_all["label"] == "maintained").sum()
        print(f"Combined: {n_syc_total} sycophantic, {n_maintained_total} maintained")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
