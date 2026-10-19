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
| bartowski IQ2_XXS (GGUF) | 170.8 GB | 1.085 | 75.21 % |

**Verdict.** Against bartowski's IQ2_XXS GGUF (170.8 GB, 11% larger), Wrench is closer to the FP8 model on all 92 windows (KL 0.922 vs 1.085; 95% CI of the difference +0.138 to +0.189) with 2.8 points higher top-1; bartowski's file gives the actual next token higher probability on the three agentic-coding windows of six full-vocabulary windows. Against the best other public quant at this size, the result is mixed: Wrench is 4.1% closer by KL (95% CI -0.052 to -0.027), all on agentic-coding windows; on general text the other file is slightly closer, and it gives the actual next token higher probability (NLL 0.085 lower, 6/6). Repeated measurements were identical. KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

FP8 reference, 92 × 4096-token windows; not task accuracy.
Hardware/runtime for the quality measurement is not recorded in the published JSON.
The speed receipt above does not establish that provenance.
