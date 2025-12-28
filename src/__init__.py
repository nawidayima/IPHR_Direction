"""MATS Rationalization Detection - Phase 1: Arcuschin Replication."""

from .data_generation import (
    Domain,
    LOCATION_PAIRS,
    DATE_PAIRS,
    POPULATION_PAIRS,
    generate_question_pairs,
    generate_geography_pairs,
    generate_date_pairs,
    generate_population_pairs,
    generate_all_pairs,
    format_chat_prompt,
    get_system_prompt,
    SYSTEM_PROMPTS,
)
from .labeling import (
    Classification,
    extract_yes_no,
    detect_contradiction,
    label_trajectory,
    calculate_contradiction_rate,
    format_for_csv,
)

__all__ = [
    # Data generation
    "Domain",
    "LOCATION_PAIRS",
    "DATE_PAIRS",
    "POPULATION_PAIRS",
    "generate_question_pairs",
    "generate_geography_pairs",
    "generate_date_pairs",
    "generate_population_pairs",
    "generate_all_pairs",
    "format_chat_prompt",
    "get_system_prompt",
    "SYSTEM_PROMPTS",
    # Labeling
    "Classification",
    "extract_yes_no",
    "detect_contradiction",
    "label_trajectory",
    "calculate_contradiction_rate",
    "format_for_csv",
]
