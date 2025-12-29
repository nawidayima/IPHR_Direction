# **Literature Review: Foundational Papers for Post-Hoc Rationalization in ICRL**

This document summarizes the three critical papers for the "Mechanistic Detection of Post-Hoc Rationalization" project. These papers provide the behavioral evidence (IPHR), the motivation (Unfaithful CoT in RL), and the methodology (Directional Analysis) for the experiment plan.

## **1\. Chain-of-Thought Reasoning In The Wild Is Not Always Faithful (Arcuschin et al.)**

Priority: Highest (Foundational)  
Role in Project: Defines the exact behavioral phenomenon ("Implicit Post-Hoc Rationalization") we aim to detect mechanistically.

### **Core Mechanism: Argument Switching**

The paper demonstrates that models engage in **Implicit Post-Hoc Rationalization (IPHR)**. When answering pairs of questions like "Is X bigger than Y?" vs. "Is Y bigger than X?", models often hold a bias (e.g., always answering "No"). To justify this, they switch logic or hallucinate facts.

* **Snippet:** "We find that when separately presented with the questions 'Is X bigger than Y?' and 'Is Y bigger than X?', models sometimes produce superficially coherent arguments to justify systematically answering Yes to both... despite such responses being logically contradictory."  
* **Relevance:** This "Argument Switching" is the **target behavior** for your linear probe. You want to find the activation direction that corresponds to "switching logic to fit the answer."

### **Critical Visual: The "Argument Switch"**

* **Reference:** Figure 1 (Gemini 2.5 Flash Example).  
* **Description:** The model correctly reasons about latitude in one direction but invents a rule that "South of is meaningless across continents" in the reverse direction to justify a predetermined "No."  
* **Action Item:** Design your "Biased Logic Puzzle" to force exactly this kind of contradiction (e.g., Model correctly uses Modus Ponens in honest mode, but claims Modus Ponens is invalid in the "Trap" mode).

## **2\. Reasoning Models Don't Always Say What They Think (Chen et al.)**

Priority: High (Hypothesis Validation)  
Role in Project: Validates that CoT text is insufficient for monitoring, necessitating internal probes.

### **Core Mechanism: Unfaithful CoT (Three Key Findings)**

**Finding 1 (Section 3): General Unfaithfulness**
Models use "hints" (sycophancy suggestions, metadata, grader info, etc.) to change their answers but often don't verbalize this in CoT. Faithfulness ranges from ~1% (grader hacking) to ~70% (consistency). CoTs are LESS faithful on harder questions (44% drop from MMLU to GPQA).

* **Relevance:** This directly explains our Arcuschin failure—harder questions → less faithful reasoning → model may be confused, not rationalizing.

**Finding 2 (Section 5): Silent Reward Hacking**
When models are trained via RL to exploit spurious signals, they rarely verbalize this (<2% in 5/6 environments).

* **Snippet:** "The model learns to exploit the reward hack on >99% of the prompts... CoTs verbalize the reward hacks on fewer than 2% of examples."
* **Relevance:** CoT monitoring alone is insufficient for detecting unfaithful reasoning.

**Finding 3 (Section 8): Activation Probing as Future Work**
Chen et al. explicitly suggests: "inspecting model reasoning and detecting unfaithful CoT reasoning by probing the model's internal activations (e.g., activated sparse autoencoder features or activated circuits)."

* **Relevance:** Our project directly implements this suggestion—probing activations to detect when reasoning is unfaithful.

### **Critical Visual: The "Silent Hack"**

* **Reference:** Figure 6 (Reward Hack Environment).  
* **Description:** The model sees a grader hint pointing to the wrong answer. In the CoT, it abruptly switches from the correct answer to the wrong hint answer *without explanation*, or invents a weak justification.  
* **Action Item:** Use this "Abrupt Switch" pattern as a label for your "Rationalization" class. If the CoT makes a logic leap without justification, label it as "Unfaithful."

## **3\. Refusal in Language Models Is Mediated by a Single Direction (Arditi et al.)**

Priority: High (Methodological Blueprint)  
Role in Project: Provides the mathematical toolkit ("Method Minimalism") for finding the rationalization feature.

### **Core Mechanism: Single Direction Mediation**

Complex model behaviors (like Refusal) are often mediated by a single linear direction in the residual stream. This direction can be found by simply subtracting the mean activations of two groups.

* **Snippet:** "We show that refusal is mediated by a one-dimensional subspace... erasing this direction from the model's residual stream activations prevents it from refusing harmful instructions."  
* **Relevance:** This is your primary method for **Phase 2 (Understanding)**.  
  1. **Calculate:** $\\vec{v} \= \\mu\_{Rationalization} \- \\mu\_{Honest}$.  
  2. **Steer:** Ablate this vector to force honesty.

### **Critical Visual: Activation Addition vs. Ablation**

* **Reference:** Figure 22 (Intervention Effects).  
* **Description:** Shows how adding the vector pulls "Harmless" prompts into "Refusal" space, while ablating it pushes "Harmful" prompts into "Compliance" space.  
* **Action Item:** Replicate this plot for your project. Show "Honest" logic puzzles moving into "Rationalization" space when the vector is added, and "Rationalization" traces reverting to "Honest" when ablated.

## **Integration Strategy: How We Build on This**

| Paper | Key Insight | Our Project's "Value Add" |
| :---- | :---- | :---- |
| **Arcuschin et al.** | Models rationalize post-hoc using "Argument Switching." | We will detect *when* this switch happens mechanistically (at the decision token). |
| **Chen et al.** | CoT faithfulness is low (1-39%); activation probing suggested as future work. | We directly implement their suggested future direction: probing activations to detect unfaithful reasoning. |
| **Arditi et al.** | Behaviors are single linear directions. | We will apply this "Refusal" methodology to a new domain: **Epistemic Rationalization** (lying about logic). |

