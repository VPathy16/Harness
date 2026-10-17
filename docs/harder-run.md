# Six-number challenge run

Three fixed tasks were selected before execution, extending the earlier four-number smoke tests. All results are retained. The rules permit fractional and negative intermediate values, with each supplied number used exactly once and only binary addition, subtraction, multiplication, division and parentheses. These are not integer-only Countdown rules.

Search alone and Gemma plus search each solved 3/3; Gemma alone solved 0/3. Every accepted expression passed the separate AST verifier using exact rational arithmetic. Search alone was faster. No larger model was tested, no cost advantage was established, and three hand-selected arithmetic tasks do not establish general reasoning ability. Laya, EmbeddingGemma and TimesFM were not used in this run.

Each mode had a 60-second per-task limit. Search had a 200,000 expansion limit. Direct and hybrid used the same Gemma Q4 model, seed 7, structured output, 256 output-token cap and thinking disabled. The small direct baseline is configuration-specific, not an estimate of Gemma's best possible performance. No exact-result cache was read. Cold worker loading is recorded separately.

Reproduce search:

```bash
python -m thinking_runtime bench --tasks examples/harder-arithmetic.json --mode symbolic --seconds 60 --expansions 200000 --output reports/harder-symbolic.json
```

Reproduce direct and hybrid, supplying the path to your llama.cpp server:

```bash
python scripts/run_local_comparison.py --server /path/to/llama-server --gguf models/gemma-gguf/gemma-4-E2B.Q4_K_M.gguf --tasks examples/harder-arithmetic.json --report-prefix harder --max-tokens 256 --seconds 60 --expansions 200000
```

See `reports/harder-summary.json` for aggregate results and `reports/harder-*.json` for expressions, model proposals, verifier feedback, times and search counts. The next valid superiority test requires a named larger-model endpoint on the identical tasks, with both model-only and tool-assisted conditions clearly stated.
