# Ling-3.1-flash — Wrench EXL3 2.10 bpw

153.6 GB EXL3 quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash).
Weights: [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw).
Runs on a GPU with most routed experts computed on the CPU from system RAM.

## Speed

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 351.2 | 12.34 | 23.33 |
| 8K | code | 347.6 | 12.21 | 23.57 |
| 64K | prose | 345.8 | 7.72 | 189.51 |
| 64K | code | 348.0 | 7.67 | 188.31 |
<!-- BENCH:END -->

Measured 2026-10-18 on one R9700 32 GB, Ryzen 7 9800X3D, 192 GB DDR5;
Linux/ROCm 7.2.1, exllamav3 `0f8cf2c3` plus the [ROCm patch](../bench/exllamav3-rocm-smem.patch).
Prefill = prompt tokens / time to first token (reading the prompt); decode = generated tokens after the first / their time
(writing the answer). Greedy, 256 tokens, no speculation or prefix reuse; two measured runs after one excluded warm-up.
The table uses the quiet rerun in [results.json](results/results.json).
Per-cell spreads and conditions: [REPRODUCE](../docs/REPRODUCE.md#reading-the-results).

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 32 GB (gfx1201); AMD Ryzen 7 9800X3D; 192 GB DDR5 |
| Software | Ubuntu 24.04.5 LTS; kernel 7.0.0-28-generic; ROCm 7.2.1 (HIP 7.2.53211); PyTorch 2.9.1+rocm7.2.1.gitff65f5bc; exllamav3 0f8cf2c3f052a2503e98c1e77d40b2f055047bcc (+local diff sha256 e9a50674a7ef9a1f728a75cdc7311ae8a65f954887a5109c3aaa55b4787dc439) |
| Run | CPU-MoE layers / threads / chunk 60 (8K) or 62 (64K) / 8 / 4,096; env `CUDA_VISIBLE_DEVICES=1`, `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib`, `OMP_NUM_THREADS=8`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `PYTORCH_ROCM_ARCH=gfx1201`; peak VRAM / RAM 30.9 GiB / 133.1 GiB |
<!-- RECEIPT:END -->

Memory peaks are GiB despite the JSON's legacy `_gb` keys. VRAM is the torch allocator peak;
host usage is system-wide, not process RSS. Loading can peak higher.

## Quality

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.922 | 77.99 % |
| Best other public quant at this size (as of 2026-10-18) | 153.6 GB | 0.962 | 77.66 % |

**Verdict.** Mixed result; neither file is better overall. By KL, Wrench is 4.1% closer to the FP8 model (95% CI -0.052 to -0.027), all of it on the agentic-coding windows; on general text the other quant is slightly closer. The other quant also gives the actual next token higher probability on six full-vocabulary windows (NLL 0.085 lower, 6/6). Repeated measurements were identical. KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

Agreement with the FP8 base on 92 × 4096-token windows; KL and top-1 are not task accuracy.
The [quality JSON](results/quality.json) does not record hardware/runtime, so the speed receipt does not describe that measurement.

## Run it

**Known ROCm issue:** this torch build lacks `shared_memory_per_block`, so the pinned runtime
fails at first decode without the [published patch](../bench/exllamav3-rocm-smem.patch). Apply it as described in INSTALL.

First follow [INSTALL](../docs/INSTALL.md) for the pinned runtime, wheels, patch, environment and weights.
**Untested interactive launch:** flags were checked; chat generation was not rerun in verification.

```sh
python "$EXL3/examples/chat.py" -m "$MODEL" -mode ling -no_think -temp 0 -mcl 60 -mct 8 -gcs 4096 -cs 9216
```

This is the measured 8K placement. At 64K use 62 CPU-MoE layers and cache size 66560.
Other GPUs and settings: [INSTALL](../docs/INSTALL.md). With a coding agent: [AI_SETUP](../docs/AI_SETUP.md).
Problems: [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md).

## Reproduce

[REPRODUCE](../docs/REPRODUCE.md) has the pinned inputs, checksum check, exact matrix and reporting rules.
Raw results and receipts: [results/](results/README.md). Harness details: [bench/](../bench/README.md).

## Limits

- One Linux/ROCm machine measured; CUDA and other GPUs untested. The shared-memory fallback is tested only on gfx1201.
- Default greedy outputs can differ across repetitions. A repeatability screen's matching pairs do not guarantee token identity.
- Chunk 8192 failed with non-finite logits on the measured stack; use 4096.

## Credits

inclusionAI for Ling-3.1-flash (MIT); [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT) for conversion/runtime;
benchmark text: PG-19 (DeepMind, Apache-2.0), llama.cpp (MIT). [NOTICE](../NOTICE).
