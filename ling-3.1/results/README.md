# Results

Rendered from `ling-3.1/results/results.json` and
`ling-3.1/results/quality.json` by `tools/fill_results.py`. The JSON is the source of truth; do not edit the marker
blocks by hand.

## One R9700 — Ling-3.1 Wrench EXL3 2.10 bpw

Raw continuation, greedy, 256 generated tokens, two measured repetitions after an
excluded warm-up. Prefill is prompt tokens divided by time to first token; decode
excludes the first token.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 377.7 | 11.98 | 21.70 |
| 8K | code | 387.5 | 11.95 | 21.10 |
| 64K | prose | 383.1 | 7.83 | 171.10 |
| 64K | code | 388.7 | 8.15 | 168.60 |
<!-- BENCH:END -->

The runner records both repetitions per cell in `cells[].reps`, plus peak VRAM
and peak RSS, the exact command and the input hashes.

## Quality against the FP8 reference

Next-token distribution on 92 x 4K windows. Lower KL and higher top-1 are better.

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.922 | 77.99 % |
| Best other public quant at this size (as of 2026-10-18) | 153.6 GB | 0.962 | 77.66 % |

**Verdict.** Mixed result; neither file is better overall. By KL, Wrench is 4.1% closer to the FP8 model (95% CI -0.052 to -0.027), all of it on the agentic-coding windows; on general text the other quant is slightly closer. The other quant also gives the actual next token higher probability on six full-vocabulary windows (NLL 0.085 lower, 6/6). Deterministic scorer (identical repeats). KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

## Inputs

`results.json` records the four input files, their token counts and their SHA-256
values. Rebuild them with `bench/build_inputs.py`; the committed
`bench/inputs/SHA256SUMS` is filled by the benchmark run.

## Render

```sh
python tools/fill_results.py --profile 3.1
python tools/fill_results.py --profile 3.1 --check
```

`--check` exits non-zero if any marker block still contains a placeholder.
