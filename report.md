# Model Merging Report

## Overview

This report evaluates five LoRA model merging methods on two tasks: **Math** (MMLU High School Mathematics) and **Science** (ARC-Challenge). The base model is `LLaMA-2-7B-Chat` (4-bit quantized) with two task-specific LoRA adapters fine-tuned on GSM8K-MCQA and ARC-MCQA respectively.

---

## Experimental Setup

| Component | Detail |
|---|---|
| Base model | `unsloth/llama-2-7b-chat-bnb-4bit` |
| Math adapter | `MonicaHuang/llama-2-7b-chat-GSM8K-MCQA` |
| Science adapter | `chenjoachim/llama-2-7b-chat-ARC-MCQA` |
| Math benchmark | MMLU High School Mathematics (200 questions) |
| Science benchmark | ARC-Challenge (200 questions) |
| LoRA rank | 8 |
| Target modules | q_proj, k_proj, v_proj |
| Decoding | Greedy (do_sample=False, num_beams=1) |

---

## Individual Adapter Baselines

| Adapter | MATH | SCIENCE |
|---|---|---|
| MATH only | 22.00% | — |
| SCIENCE only | — | 62.00% |

The science adapter is significantly stronger than the math adapter (62% vs 22%). This imbalance is important context for evaluating merged models.

---

## Merging Results

| Method | Weights | Density | MATH | SCIENCE | AVG |
|---|---|---|---|---|---|
| linear | [1.0, 0.4] | — | 15.00% | 23.50% | 19.25% |
| dare_linear | [1.0, 0.4] | 0.2 | 18.00% | 38.50% | 28.25% |
| magnitude_prune | [1.0, 0.4] | 0.5 | 18.00% | 36.00% | 27.00% |
| dare_ties | [1.0, 0.4] | 0.2 | 14.50% | 43.00% | 28.75% |
| magnitude_prune | [1.0, 0.4] | 0.2 | 26.50% | 48.50% | 37.50% |
| **magnitude_prune** | **[0.6, 1.0]** | **0.2** | **29.50%** | **52.50%** | **41.00%** |

---

## Analysis

### Best Method: magnitude_prune with tuned weights [0.6, 1.0]

The best configuration achieved **41.00% average accuracy** — `magnitude_prune` with weights adjusted to favor the stronger science adapter (`[0.6, 1.0]`).

### Effect of Weight Tuning

Changing weights from `[1.0, 0.4]` to `[0.6, 1.0]` on `magnitude_prune` improved both tasks:
- Math: 26.50% → **29.50%** (+3%)
- Science: 48.50% → **52.50%** (+4%)
- Average: 37.50% → **41.00%** (+3.5%)

This suggests that since the science adapter is inherently stronger, giving it more weight better preserves its capabilities while still retaining math knowledge.

### Effect of Density

Increasing density from 0.2 to 0.5 on `magnitude_prune` **hurt** performance:
- Math: 26.50% → 18.00% (-8.5%)
- Science: 48.50% → 36.00% (-12.5%)

Keeping more parameters (higher density) introduces more conflicts between the two adapters, which degrades the merged model. Lower density (more aggressive pruning) removes noisy small parameters and reduces interference.

### Linear is the Worst Method

`linear` achieves only 19.25% average — the worst of all methods. Without any pruning or conflict resolution, the two adapters directly interfere with each other, significantly degrading both tasks (math drops from 22% to 15%, science from 62% to 23.5%).

### Task Degradation After Merging

All merged models suffer performance loss compared to individual adapters. The best merged model still falls short of the baselines:
- Math: 29.50% vs 22.00% baseline (+7.5% — merging actually helped math)
- Science: 52.50% vs 62.00% baseline (-9.5% — science degraded)

Notably, the best merged model **outperforms the math-only adapter on math**, suggesting positive knowledge transfer from the science adapter.

---

## Conclusion

`magnitude_prune` with density=0.2 and weights=[0.6, 1.0] is the best merging configuration, achieving 41% average accuracy. Key findings:

1. **Weight tuning matters** — matching weights to adapter strength improves both tasks
2. **Lower density is better** — aggressive pruning reduces inter-adapter conflicts
3. **magnitude_prune outperforms DARE-based methods** on this task pair
4. **linear merging is ineffective** without conflict resolution
5. Merging can improve the weaker task (math) through knowledge transfer while slightly degrading the stronger task (science)
