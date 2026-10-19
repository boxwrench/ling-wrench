# Results

Source files: [results.json](results.json) (quiet speed rerun, 2026-10-18), [quality.json](quality.json) (FP8 agreement).
Rendered by [fill_results.py](../../tools/fill_results.py); never edit marker blocks by hand.

## Speed

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 486.6 | 12.02 | 16.84 |
| 8K | code | 498.7 | 12.07 | 16.43 |
| 64K | prose | 476.5 | 7.86 | 137.53 |
| 64K | code | 476.4 | 7.64 | 137.58 |
<!-- BENCH:END -->

R9700 32 GB, Ryzen 7 9800X3D, 192 GB DDR5; Ubuntu 24.04.5, ROCm 7.2.1,
PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, exllamav3 `0f8cf2c3` plus the published ROCm patch.
Greedy raw continuation; 256 generated tokens, no MTP or prefix reuse; two measured runs after one excluded warm-up.
The JSON retains repetitions, hashes, actual/cached counts and memory receipts.
Definitions, repeat spread and commands: [REPRODUCE](../../docs/REPRODUCE.md).

## Quality

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.922 | 77.99 % |
| Best other public quant at this size (as of 2026-10-18) | 153.6 GB | 0.962 | 77.66 % |

**Verdict.** Mixed result; neither file is better overall. By KL, Wrench is 4.1% closer to the FP8 model (95% CI -0.052 to -0.027), all of it on the agentic-coding windows; on general text the other quant is slightly closer. The other quant also gives the actual next token higher probability on six full-vocabulary windows (NLL 0.085 lower, 6/6). Repeated measurements were identical. KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

FP8 reference, 92 × 4096-token windows; not task accuracy.
Hardware/runtime for the quality measurement is not recorded in the published JSON.
The speed receipt above does not establish that provenance.
