# Natural-language compiler: measured outcome

End-to-end success: **1/3**. The only pass was detecting an impossible ranking problem. Neither satisfiable problem passed.

| Task | Outcome | Compilation time |
|---|---|---:|
| six-job-schedule | Mistranslated adjacency and added unsupported fixed slots; wrong final schedule | 78.09 s |
| robot-bays | Rejected before execution: eq/neq constraint had the wrong number of variables | 88.45 s |
| cyclic-ranks | Correct translation; solver proved unsatisfiable | 12.54 s |

Times are single CPU runs, not statistical latency estimates. They exclude model loading and offline semantic audit. The declared cap was 1,800 output tokens and 120 seconds per task; thinking was disabled.

## Concrete failure

The scheduling prompt required B immediately before E. The model instead used a weaker B < E constraint and invented fixed slots. Its answer put B in slot 2 and E in slot 5. That answer satisfied the model-generated IR but violated the original problem. The withheld benchmark reference detected the error. Ordinary execution verification against generated IR alone would not have detected the mistranslation.

## Concrete success

For ranks R, S, T, the model correctly encoded all-different plus R < S, S < T, T < R, with domains 1..3. Generic search exhausted the possibilities and reported unsatisfiable. Execution took 0.0324 ms after 12.54 s of compilation.

## Interpretation

The reusable compiler-to-solver path runs, but this tested Gemma setup is unreliable at semantic translation. There is no demonstrated general-purpose planner, no demonstrated advantage over a larger model, and no cost advantage measurement. The positive result is limited to one newly authored contradictory instance in a predefined constraint language.

All three cases and the solver were fixed before model evaluation. No retries, gold-informed fixes, or reference substitutions were made. The complete original model outputs, including the rejected robot output, remain in `compiler-run.json`. All 20 unit tests passed.

For reproduction and evaluation boundaries, see `../docs/compiler-experiment.md`.
