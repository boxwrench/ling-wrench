# Results

Rendered from `results/results.json` and `results/quality.json` by
`tools/fill_results.py`. The JSON is the source of truth; do not edit the marker
blocks by hand.

## One R9700 — Ling-3.1 Wrench EXL3 2.10 bpw

Raw continuation, greedy, 256 generated tokens, two measured repetitions after an
excluded warm-up. Prefill is prompt tokens divided by time to first token; decode
excludes the first token.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | TBD | TBD | TBD |
| 8K | code | TBD | TBD | TBD |
| 64K | prose | TBD | TBD | TBD |
| 64K | code | TBD | TBD | TBD |
<!-- BENCH:END -->

The runner records both repetitions per cell in `cells[].reps`, plus peak VRAM
and peak RSS, the exact command and the input hashes.

## Quality against the FP8 reference

Next-token distribution on 92 x 4K windows. Lower KL and higher top-1 are better.

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.925 | 77.99 % |
| turboderp EXL3 2.10 bpw | 153.6 GB | 0.961 | 77.70 % |
| AtomicChat AD-IQ2_XXS | 174.1 GB | 1.071 | 75.68 % |

**Provisional.** Single runs; KL is not task accuracy. The same-config repeat floor is still being measured.
<!-- QUALITY:END -->

## Inputs

`results.json` records the four input files, their token counts and their SHA-256
values. Rebuild them with `bench/build_inputs.py`; the committed
`bench/inputs/SHA256SUMS` is filled by the benchmark run.

## Render

```sh
python tools/fill_results.py
python tools/fill_results.py --check
```

`--check` exits non-zero if any marker block still contains a placeholder.
