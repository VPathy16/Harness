# Natural-language compiler experiment

This experiment tests a different capability from the manually encoded river puzzle: a small model must translate an English problem into data that a task-independent solver can execute.

## Pipeline

1. Give Gemma a fixed integer-constraint language, one unrelated worked example, and an English problem.
2. Constrain the response to a JSON schema. Schema validity does not establish semantic correctness.
3. Validate references, domains, operator arities and bounds.
4. Execute only the model-produced IR using generic finite-domain backtracking.
5. Independently check any assignment against the compiled constraints.
6. For benchmark evaluation only, compare the model's feasible solution set against a withheld human-authored reference and validate its answer against that reference.

No reference rules, solutions, evaluation feedback or task-specific Python are supplied to the model. No reference IR is substituted into the execution path. There is one model call per case, without repair. All successes and failures are retained.

## Scope and controls

The supported operators are `all_different`, `eq`, `neq`, `lt`, `offset`, and `sum_eq`. Variables have explicitly enumerated integer domains. This language supports a limited family of assignment, ordering and arithmetic constraints; it does not represent arbitrary programs or arbitrary reasoning.

The three cases cover six-job scheduling, robot-to-bay assignment, and an inconsistent rank ordering. These are newly authored instances in this experiment, not evidence that the underlying task families are absent from model training. The solver and case file were hashed before the first model request; hashes are in `reports/compiler-freeze.json` and checked by the runner.

Model settings: the existing Gemma E2B Q4_K_M checkpoint, temperature zero, seed 7, thinking disabled, 1,800 output tokens per call, 120 seconds for compilation and execution per task. Model loading and offline semantic audit time are reported separately. No financial cost is inferred from CPU time. There is no larger-model baseline in this experiment.

## What counts as success

An end-to-end pass requires semantic equivalence of the feasible assignment sets and either a reference-valid assignment or a correctly established absence of solutions. Equivalent constraints may be written in a different order or form. For inconsistent problems, equivalence of empty solution sets does not prove faithful preservation of each individual textual rule.

The reference checker is an evaluation instrument. In deployment, an arbitrary new English task has no automatic gold specification. The runtime can verify execution against compiled rules, but it cannot thereby prove that the rules captured the user's intent. Human review or another independently grounded source of constraints is still needed for that guarantee.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/run_compiler.py --server /path/to/llama-server --gguf models/gemma-gguf/gemma-4-E2B.Q4_K_M.gguf
```

`reports/compiler-run.json` records exact prompts, model output, generated IR, execution, semantic audits and timing. `reports/compiler-results.md` summarizes the measured run. The other specialist models and reinforcement learning are not used here.
