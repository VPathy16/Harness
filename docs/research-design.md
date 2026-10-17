# Research claim and test design

## Claim

For a declared task distribution and deployment budget, can a small-model runtime
achieve higher independently verified task success, lower end-to-end latency, and
lower amortized cost than a configured GPT-OSS-120B system?

This is a system comparison. It is not a parameter-count proof, and arbitrary task
families do not inherit the guarantees of the arithmetic domain.

## Conditional theory

For solvable tasks with decidable correct verification and a finite candidate
grammar containing a solution, complete enumeration eventually finds a solution.
Neural ranking may change order, but must not permanently prune valid candidates
if completeness is claimed. This says nothing useful about practical time until a
bound on search effort is established. The arithmetic implementation merges
value-equivalent states because future reachability depends only on remaining
fraction values; this equivalence is not valid for all domains.

If, conditional on every unsuccessful history, the probability of a verified
solution on the next attempt is at least q, and the verifier has no false accepts,
then P(success by N) >= 1-(1-q)^N. This conditional lower bound is stronger than
measuring an average first-attempt success rate. Correlated failures can invalidate
it. For a baseline success probability p under a specified budget, the bound exceeds
p when N > log(1-p)/log(1-q), for 0 < p,q < 1.

If a full attempt costs at most c and takes at most t, with setup overhead C0,T0,
a sufficient three-axis condition is:

    1-(1-q)^N > p
    C0 + N*c < C_baseline
    T0 + N*t < T_baseline

The current model calls have cooperative rather than hard deadlines; do not apply
the strict latency bound until kernel execution has a proven limit or worker-level
cancellation. A predicted value from a learned critic is not a sound verifier.
Neither the model-size table nor vendor latency numbers establish q, c or t here.

## Experimental controls

1. Freeze a fresh test distribution before tuning. Separate calibration/train/dev/test.
2. Log checkpoint hashes, software versions, hardware and model-serving settings.
3. Compare small direct, small feedback, small hybrid, symbolic-only, Laya-guided
   search, large feedback and large hybrid. Give every baseline the same public
   specification and verification access. Direct one-shot is only an ablation.
4. Report paired success outcomes with uncertainty, not just a tiny-suite percentage.
5. Report warm and cold latency, p50/p95, failures/timeouts, complete per-task cost,
   peak client/worker memory and server/GPU memory separately.
6. Include all model/token/solver/search work, not just successful generation calls.
7. Account for training and setup costs amortized over a declared task count.
8. Evaluate retrieval/forecasting on tasks that actually need those capabilities.
9. For transfer, train on one set of compositions and test new compositions. Exact
   memorization and retrieval of test answers must not count as learned abstraction.

## Next research implementation

- Add a second executable domain with an independent verifier (e.g. small scheduling
  problems). Require a typed task specification first; then measure language-to-spec
  correctness separately instead of assuming the compiler always understands intent.
- Add a grammar of reusable procedures and learn candidate-order/value estimates
  from verified traces. A cached answer is not a new reasoning procedure.
- Compare a bandit/policy learner against deterministic search before adding MCTS.
- Use actual transition execution as the simulator when possible. TimesFM forecasts
  observed numeric histories; it is not the substitute for this transition model.
- Add hard worker cancellation and process-isolated tool execution before permitting
  arbitrary code or any user-system modifications.

## Primary references

- Neural-Guided Deductive Search: https://arxiv.org/abs/1804.01186
- Neural-guided bidirectional program search: https://arxiv.org/abs/2110.11536
- PAL: https://proceedings.mlr.press/v202/gao23f
- MuZero: https://arxiv.org/abs/1911.08265
- Gemma E2B: https://huggingface.co/google/gemma-4-E2B
- Laya: https://huggingface.co/convaiinnovations/laya
- EmbeddingGemma 2: https://huggingface.co/google/embeddinggemma-2
- TimesFM 2.5: https://huggingface.co/google/timesfm-2.5-200m-pytorch
- GPT-OSS architecture: https://openai.com/index/introducing-gpt-oss/
