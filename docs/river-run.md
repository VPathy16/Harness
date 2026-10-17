# Five-pair river crossing: measured run

Task: five actor-agent pairs, boat capacity three. Everyone starts on the left. Each crossing must carry one to three people. An actor may be with another agent only if their own agent is present. Safety applies inside the boat, on both banks during departure, and after arrival.

## Results

| Configuration | Result | Time excluding model loading |
|---|---|---:|
| Breadth-first search | Verified shortest solution, 11 crossings | 0.004268 s |
| Gemma E2B Q4 alone | Rejected | 7.455 s |
| Same Gemma proposal + search fallback | Verified | 7.459 s |

Search expanded 173 states, considered 6960 candidate crossings, and visited 179 unique states. Independent verification uses sets; search uses bitsets. Breadth-first traversal with unit-cost crossings establishes minimal crossing count. No solution sequence is embedded or read from cache.

Gemma failure: Move 2: passenger absent from departure bank. After sending a1 and A1 right, it attempted to return a1 and A2, although A2 was still left.

Gemma settings: seed 7, temperature 0.7, Q4_K_M, thinking disabled, structured JSON, maximum 2,048 output tokens, 120-second task budget. The response finished before the limit. Hybrid reuses the direct proposal and timing rather than issuing a second model call. Cold loading took 2.776 seconds.

## Verified solution

| Crossing | Direction | Passengers |
|---:|---|---|
| 1 | L->R | a1, a2 |
| 2 | R->L | a1 |
| 3 | L->R | a1, a3, a4 |
| 4 | R->L | a1 |
| 5 | L->R | A2, A3, A4 |
| 6 | R->L | a2, A2 |
| 7 | L->R | A1, A2, A5 |
| 8 | R->L | a3 |
| 9 | L->R | a1, a2 |
| 10 | R->L | a1 |
| 11 | L->R | a1, a3, a5 |

The complete bank state after each crossing is recorded in `reports/river-run.json`.

## What this establishes

The new manually implemented river domain plus search solves this concrete task. The small model did not contribute a valid plan. Laya, EmbeddingGemma, TimesFM, learned routing, reinforcement learning and automatic compilation of prose rules were not used. This is a verified solver extension, not evidence of general intelligence or a learned general planning compiler. CPU wall time is measured; financial cost is not.

## Published reference, separate from local results

The authors of *Rethinking the Illusion of Thinking* report that Gemini 2.5 Pro failed every trial at N=5, k=3; their paper specifies ten trials per configuration. This is external reported evidence under their prompts and settings. Gemini was not run locally, its historical API snapshot was not reproduced, and this is not a matched contemporary large-model comparison. A larger model given a comparable search tool may also solve it.

- Paper: https://arxiv.org/abs/2507.01231
- Authors' report: https://github.com/11inaki11/Rethinking-The-Illusion-of-Thinking
- Original benchmark rules (Appendix A.1.3): https://papers.nips.cc/paper_files/paper/2025/file/9b26ad15462c81548c0689188d2e8018-Paper-Conference.pdf

## Reproduce

Search only:

```bash
python scripts/run_river.py
```

Include local Gemma:

```bash
python scripts/run_river.py --server /path/to/llama-server --gguf models/gemma-gguf/gemma-4-E2B.Q4_K_M.gguf
```

Validation: all 14 unit tests passed, including boat safety, bank safety, illegal passengers, known shortest solutions, infeasible cases, budget exhaustion and corrupted plans.
