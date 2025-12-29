# Findings Report: Detecting Post-Hoc Rationalization in Llama-3-8B

**Date:** 2024-12-28
**Hours 0-9 Summary**

---

## Executive Summary

We attempted to find a linear direction in Llama-3-8B's residual stream that distinguishes "rationalizing" responses from "honest" responses using Arcuschin et al.'s paired geographic questions paradigm.

**Primary Result:** No detectable linear signal (AUC ~0.45-0.50 across all layers and methods).

**Key Discovery:** Contradiction rate strongly correlates with question difficulty (45% easy → 80% hard), suggesting the model's contradictions stem from **genuine uncertainty** about geographically close cities rather than **post-hoc rationalization** of predetermined conclusions.

---

## Methodology

### Data Generation (Hours 0-6)
- **Task:** Paired geographic questions ("Is X south of Y?" / "Is Y south of X?")
- **Model:** Llama-3-8B-Instruct
- **Labeling:** Contradiction = model answers NO/NO or YES/YES (logically impossible)
- **Result:** 50 geography pairs, 30 contradictions (60%), 20 honest

### Activation Extraction (Hours 6-8)
- **Token position:** Last token of prompt (before generation)
- **Layers:** 24, 28, 31 (upper third of 32-layer model)
- **Shape:** [n_samples, 4096] per layer

### Probe Training (Hours 8-9)
- **Methods:** Difference-in-Means (DiM), Logistic Regression (L2 regularized)
- **Evaluation:** 5-fold stratified cross-validation
- **Focus:** Geography domain only, Question B only (where "rationalization" would occur)

---

## Results

### Probe Performance

| Layer | DiM AUC | Best LR AUC | Interpretation |
|-------|---------|-------------|----------------|
| 24 | 0.442 ± 0.090 | 0.475 ± 0.086 | No signal |
| 28 | 0.467 ± 0.122 | 0.467 ± 0.116 | No signal |
| 31 | 0.492 ± 0.096 | 0.475 ± 0.086 | No signal |

All results are statistically indistinguishable from random (0.5).

### Contradiction Rate by Difficulty

| Difficulty | Contradiction Rate | Example Pairs |
|------------|-------------------|---------------|
| Easy | 45% (9/20) | Paris/Cairo, Tokyo/Sydney |
| Medium | 60% (9/15) | New York/Mexico City, Mumbai/Singapore |
| Hard | **80%** (12/15) | Seattle/Portland, Milan/Rome, Frankfurt/Prague |

### Answer Pattern Distribution

| Pattern | Count | Description |
|---------|-------|-------------|
| NO/NO | 29 | Model says neither city is south of the other |
| YES/YES | 1 | Model says both cities are south of each other |
| NO/YES | 15 | Correct pattern (consistent) |
| YES/NO | 2 | Correct pattern (consistent) |

---

## Interpretation

### The "Hard" Pairs Are Geographically Close

Examining the hard-difficulty contradictions reveals a pattern:

**Same region/country:**
- Seattle/Portland (Pacific Northwest, ~280km apart)
- Milan/Rome (Italy, ~480km apart)
- Boston/Washington DC (US East Coast)
- Toronto/Detroit (Great Lakes region)

**Similar latitudes in Europe:**
- Frankfurt/Prague (50.1°N vs 50.1°N - nearly identical!)
- Lyon/Venice (45.8°N vs 45.4°N)
- Manchester/Hamburg (53.5°N vs 53.6°N)
- Glasgow/Copenhagen (55.9°N vs 55.7°N)
- Zurich/Budapest (47.4°N vs 47.5°N)
- Geneva/Ljubljana (46.2°N vs 46.1°N)

### Hypothesis Revision

**Original hypothesis (H1):**
> Post-hoc rationalization is mediated by a linearly separable direction in the residual stream.

**Revised interpretation:**
The Arcuschin "contradiction" behavior may not be rationalization in the intended sense. Instead:

1. **Model uncertainty:** On geographically close pairs, the model is genuinely uncertain about relative positions
2. **Conservative default:** When uncertain, the model defaults to "NO" (the safer, more common answer)
3. **Not rationalization:** The model isn't "knowing" the right answer and suppressing it - it's genuinely confused

This is supported by:
- Strong difficulty gradient (45% → 80% contradiction rate)
- "Hard" pairs are objectively harder (similar latitudes, same regions)
- No detectable activation difference between contradiction and honest responses

---

## Implications

### For This Project

The Arcuschin paradigm may not be suitable for detecting rationalization because:
1. The contradictions appear to be driven by genuine uncertainty
2. There's no "ground truth the model knows but suppresses"
3. The behavior is more like "confusion" than "deception"

### For Interpretability Research

This is an **informative negative result**:
- Not all behavioral inconsistencies are "rationalization"
- Difficulty/uncertainty is a confound that must be controlled
- Future work should distinguish "lying" from "being wrong"

---

## Additional Test: Easy Pairs Only

To test whether "easy" contradictions (geographically distant cities like Paris/Cairo) show clearer signal than "hard" contradictions (close cities like Frankfurt/Prague), we ran a focused analysis on easy pairs only.

### Easy Pairs Dataset
- **Total samples:** 20 (Question B only)
- **Contradictions:** 9 (45%)
- **Honest:** 11 (55%)

### Results (5-fold CV)

| Layer | DiM AUC | LR AUC | Interpretation |
|-------|---------|--------|----------------|
| 24 | 0.467 ± 0.172 | 0.467 ± 0.172 | No signal |
| 28 | 0.583 ± 0.105 | 0.583 ± 0.105 | Noise (high variance) |
| 31 | 0.417 ± 0.190 | 0.467 ± 0.172 | No signal |

Layer 28's 0.583 appears slightly above random, but fold-level results [0.67, 0.75, 0.5, 0.5, 0.5] show this is driven by 2 lucky folds rather than consistent signal.

### Conclusion

Even contradictions on "easy" pairs (where the model should clearly know the answer) show no detectable activation signature. This strongly suggests:

1. **The Arcuschin contradiction behavior is not mediated by a linearly separable direction** in the residual stream at the last-token position
2. **Pivoting to ICRL (Chen et al.)** is the correct decision - reward-induced rationalization may show clearer signal since the model definitively knows the correct answer before changing it

---

## Decision Point: Next Steps

### Option A: Continue with Arcuschin (Different Token Position)

**Pros:**
- Infrastructure already built
- Might find signal at first generated token

**Cons:**
- Requires re-running activation extraction (~2 hours)
- Fundamental issue: behavior appears to be confusion, not rationalization
- Even if we find a signal, it might be an "uncertainty" direction, not "rationalization"

**Time estimate:** 3-4 hours

### Option B: Pivot to ICRL (Chen et al. Reward Hacking)

**Pros:**
- Different mechanism: model learns to give wrong answer for reward
- More clearly "rationalization" - model should know correct answer
- Tests H2 (generalization) directly

**Cons:**
- Need to set up new data generation pipeline
- Need to simulate reward feedback
- Uncertain if Llama-3-8B shows the effect

**Time estimate:** 5-6 hours

### Option C: Write Up Negative Result

**Pros:**
- Valuable contribution: "Arcuschin contradictions are confusion, not rationalization"
- Frees time for clear documentation
- Honest about what we found

**Cons:**
- No positive result to report
- Less exciting than finding a direction

**Time estimate:** 2-3 hours

---

## Recommendation

**UPDATE (after easy-pairs test):**

Easy pairs showed no signal (AUC ~0.5). The decision is clear:

**→ PIVOT TO ICRL (Chen et al. Reward Hacking)**

Rationale:
1. Arcuschin paradigm exhaustively tested: full dataset, geography-only, Question B only, easy pairs only - all null
2. ICRL provides cleaner "rationalization" signal: model gives correct answer, receives negative feedback, changes answer
3. Time budget allows: ~7 hours remaining for experimentation before distillation phase

The Arcuschin negative result is itself valuable - it demonstrates that behavioral inconsistency (contradiction) does not imply a detectable "rationalization mode" and may instead reflect genuine uncertainty.

---

## Files Generated

- `experiments/run_20251228_204835_expand_dataset/activations/residual_stream_activations.pt`
- `experiments/run_20251228_204835_expand_dataset/probes/rationalization_probes.pt`
- `experiments/run_20251228_204835_expand_dataset/probes/geography_question_b_results.pt`
- `notebooks/03_extract_activations.ipynb`
- `notebooks/04_train_probes.ipynb`
- `notebooks/05_probe_geography_only.ipynb`
- `scripts/quick_easy_pairs_test.py`

---

## Next Phase: ICRL Sycophancy (Hours 9-14)

### Revised Hypothesis

Since H1 (bias-induced rationalization) showed no signal, we cannot test H2 (generalization). Instead, we pivot to a **standalone hypothesis**:

**H1' (Revised):** Sycophantic behavior (changing a correct answer after negative feedback) is mediated by a linearly separable direction in the residual stream.

### Key Difference from Arcuschin

| Aspect | Arcuschin (failed) | ICRL Sycophancy |
|--------|-------------------|-----------------|
| Source of error | Model uncertainty? | Model knows answer is correct |
| Mechanism | Unclear | Social pressure / reward hacking |
| Ground truth | Ambiguous | Clear (model just gave correct answer) |

### Approach
1. Generate sycophancy trajectories: model answers correctly → fake negative feedback → model changes answer
2. Extract activations before the changed answer
3. Train probe: "maintaining correct answer" vs "sycophantically changing"
4. Target: ROC-AUC > 0.7