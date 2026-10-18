# Results

This page is rendered from `results/results.json` and `results/quality.json` by
`tools/fill_results.py`. The JSON is the source of truth; do not edit the
marker blocks by hand.

## One R9700 — Ling-3.1 Wrench EXL3 2.10 bpw

Raw continuation, greedy, 256 generated tokens, two measured repetitions after
an excluded warm-up. Prefill is prompt tokens divided by time to first token;
decode excludes the first token.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | TBD | TBD | TBD |
| 8K | code | TBD | TBD | TBD |
| 64K | prose | TBD | TBD | TBD |
| 64K | code | TBD | TBD | TBD |
<!-- BENCH:END -->

The runner records both repetitions per cell in `cells[].reps`, plus peak VRAM
and peak RSS. The baseline block, when present, holds the E216-style control at
8K code; the screen block holds the knob deltas from the tuning pass.

## Quality against the FP8 reference

Locked 92 × 4096-token corpus. `KL top-64` and `KL_pred` are over 92 windows;
`exact KL` and `exact dNLL` are full-vocabulary on the 6-window prefix.

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Exact KL | KL_pred | Top-1 | Exact dNLL |
|---|---:|---:|---:|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.925 | 1.161 | 0.130 | 77.99 % | -0.022 |
| turboderp EXL3 2.10 bpw | 153.6 GB | 0.961 | 1.187 | 0.130 | 77.70 % | -0.092 |
| AtomicChat AD-IQ2_XXS | 174.1 GB | 1.071 | 1.330 | 0.176 | 75.68 % | -0.062 |

**Verdict pending a noise-floor check.** The numbers above are single-run rankings; the same-configuration repeat noise floor is still being measured. Treat the ordering as provisional until it is reported.
<!-- QUALITY:END -->

The method and uncertainty rule are in [../eval/METHOD.md](../eval/METHOD.md).
The headline reading is that the Wrench calibration lowers plain KL and raises
top-1 on agentic traffic at the same size, pays a small public-prose cost, and
does not improve exact actual-token dNLL on the 6-window prefix.

## Inputs

`results.json` records the four input files, their token counts and their
SHA-256 values. Rebuild them with `bench/build_inputs.py`; the committed
`bench/inputs/SHA256SUMS` is filled by the benchmark run.

## Render

```sh
python tools/fill_results.py
python tools/fill_results.py --check
```

`--check` exits non-zero if any marker block still contains a placeholder.
