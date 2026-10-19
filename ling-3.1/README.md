# Ling-3.1-flash — Wrench EXL3 2.10 bpw

153.6 GB EXL3 quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash).
Weights: [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw).
Runs on a GPU with most routed experts computed on the CPU from system RAM.

## Speed

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 486.6 | 12.02 | 16.84 |
| 8K | code | 498.7 | 12.07 | 16.43 |
| 64K | prose | 476.5 | 7.86 | 137.53 |
| 64K | code | 476.4 | 7.64 | 137.58 |
<!-- BENCH:END -->

Measured 2026-10-18 on one R9700 32 GB, Ryzen 7 9800X3D, 192 GB DDR5;
Linux/ROCm 7.2.1, exllamav3 `0f8cf2c3` plus the [ROCm patch](../bench/exllamav3-rocm-smem.patch).
Prefill = prompt tokens / time to first token (reading the prompt); decode = generated tokens after the first / their time
(writing the answer). Greedy, 256 tokens, no speculation or prefix reuse; two measured runs after one excluded warm-up.
Settings: explicit single-device load (`--device cuda:0`), `EXL3_MOE_STREAM_T=6`, chunk 8192, 62 CPU-MoE layers,
8 threads ([results.json](results/results.json)). Against the default load (autosplit, chunk 4096: 351.2/347.6 at 8K,
345.8/348.0 at 64K) that is +38-43% prefill with unchanged decode. A six-window exact comparison with the FP8 model
puts the change inside the same-settings repeat noise (no measurable quality change).
Per-cell spreads and conditions: [REPRODUCE](../docs/REPRODUCE.md#reading-the-results).

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 32 GB (gfx1201); AMD Ryzen 7 9800X3D; 192 GB DDR5 |
| Software | Ubuntu 24.04.5 LTS; kernel 7.0.0-28-generic; ROCm 7.2.1 (HIP 7.2.53211); PyTorch 2.9.1+rocm7.2.1.gitff65f5bc; exllamav3 0f8cf2c3f052a2503e98c1e77d40b2f055047bcc (+local diff sha256 e9a50674a7ef9a1f728a75cdc7311ae8a65f954887a5109c3aaa55b4787dc439) |
| Run | CPU-MoE layers / threads / chunk 62 (8K) or 62 (64K) / 8 / 8,192; env `CUDA_VISIBLE_DEVICES=1`, `EXL3_MOE_STREAM_T=6`, `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib`, `OMP_NUM_THREADS=8`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `PYTORCH_ROCM_ARCH=gfx1201`; peak VRAM / RAM 31.2 GiB / 134.7 GiB |
<!-- RECEIPT:END -->

Memory peaks are GiB despite the JSON's legacy `_gb` keys. VRAM is the torch allocator peak;
host usage is system-wide, not process RSS. Loading can peak higher.

## Quality

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.922 | 77.99 % |
| Best other public quant at this size (as of 2026-10-18) | 153.6 GB | 0.962 | 77.66 % |
| bartowski IQ2_XXS (GGUF) | 170.8 GB | 1.085 | 75.21 % |

**Verdict.** Against bartowski's IQ2_XXS GGUF (170.8 GB, 11% larger), Wrench is closer to the FP8 model on all 92 windows (KL 0.922 vs 1.085; 95% CI of the difference +0.138 to +0.189) with 2.8 points higher top-1; bartowski's file gives the actual next token higher probability on the three agentic-coding windows of six full-vocabulary windows. Against the best other public quant at this size, the result is mixed: Wrench is 4.1% closer by KL (95% CI -0.052 to -0.027), all on agentic-coding windows; on general text the other file is slightly closer, and it gives the actual next token higher probability (NLL 0.085 lower, 6/6). Repeated measurements were identical. KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

Agreement with the FP8 base on 92 × 4096-token windows; KL and top-1 are not task accuracy.
Scored on the same R9700 with exllamav3 in its deterministic settings ([quality JSON](results/quality.json)).

## Run it

**Known ROCm issue:** this torch build lacks `shared_memory_per_block`, so the pinned runtime
fails at first decode without the [published patch](../bench/exllamav3-rocm-smem.patch). Apply it as described in INSTALL.

First follow [INSTALL](../docs/INSTALL.md) for the pinned runtime, wheels, both patches, environment and weights.

**Measured interactive path.** `examples/chat.py` loads through `model_init`. Apply the
[explicit-device patch](../bench/exllamav3-explicit-device.patch) and pass `-dev cuda:0`. On the public 8K code input
(8192 prompt tokens, greedy 256 generated, 16-token warm-up excluded, two reps) this loader measured **502.4 tok/s
prefill / 12.24 tok/s decode** (reps 500.6/12.24 and 504.2/12.24). The benchmark runner measured 505.8/12.20 on the
same box in the same session; the published fast row is 498.7/12.07. The chat console loop itself was not driven.

```sh
EXL3_MOE_STREAM_T=6 python "$EXL3/examples/chat.py" -m "$MODEL" \
  -dev cuda:0 -nw -mode ling -no_think -temp 0 \
  -mcl 62 -mct 8 -gcs 8192 -cs 9216
```

Without `-nw`, the same load alternated about 475 and 500 tok/s across 256-token repetitions; `-nw` skips
`model_init`'s post-load warm-up, so the first interactive prompt pays kernel autotune instead. Omit `-dev` for the
measured autosplit fallback (60 CPU-MoE layers, chunk 4096, about 350 tok/s prefill). `-dev` is refused with `-gs` or
`-tp`; see [INSTALL](../docs/INSTALL.md) for the flag and placement details.

For agents and showcases, [bench/serve_fast.py](../bench/serve_fast.py) loads with the same flags and serves
`POST /v1/chat/completions` on localhost only. It is non-streaming, handles one request at a time, uses the Ling chat
template with thinking off unless the request sets `"thinking": true`, and is greedy by default.

```sh
EXL3_MOE_STREAM_T=6 python bench/serve_fast.py -m "$MODEL" \
  -dev cuda:0 -nw -mcl 62 -mct 8 -gcs 8192 -cs 9216 --port 8000
```

Other GPUs and settings: [INSTALL](../docs/INSTALL.md). With a coding agent: [AI_SETUP](../docs/AI_SETUP.md).
Problems: [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md).

## Reproduce

[REPRODUCE](../docs/REPRODUCE.md) has the pinned inputs, checksum check, exact matrix and reporting rules.
Raw results and receipts: [results/](results/README.md). Harness details: [bench/](../bench/README.md).

## Limits

- One Linux/ROCm machine measured; CUDA and other GPUs untested. The shared-memory fallback is tested only on gfx1201.
- Default greedy outputs can differ across repetitions. A repeatability screen's matching pairs do not guarantee token identity.
- Chunk 8192 is measured only with the explicit loader and 62 CPU-MoE layers; under autosplit it gave non-finite logits, so use 4096 there.

## Credits

inclusionAI for Ling-3.1-flash (MIT); [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT) for conversion/runtime;
benchmark text: PG-19 (DeepMind, Apache-2.0), llama.cpp (MIT). [NOTICE](../NOTICE).
