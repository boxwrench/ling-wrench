# Ling-3.1-flash — Wrench EXL3 2.10 bpw

An EXL3 trellis quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) at 2.10 bits per weight, 153.6 GB. The weights are at [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw); this page holds the measured numbers, the exact run command and the benchmark harness shared with the Ling-3.0 releases.

Part of [boxwrench/ling-wrench](https://github.com/boxwrench/ling-wrench).

> **Status, 2026-10-18: provisional numbers.** The speed table below was measured on 2026-10-17 with our own run script on
> exllamav3 `bd4da327` plus a local patch (the RDNA K = 2 fix, since merged upstream). A re-run on clean upstream exllamav3
> with the public harness is in progress today; this page will be updated with its results.

## Base model

Ling-3.1-flash is inclusionAI's 560B-total, ~25B-active hybrid MoE. It has 63 KDA linear-attention layers and 9 gated MLA layers, an MTP head, and a 1M-token context window. It is MIT-licensed; see the [model card](https://huggingface.co/inclusionAI/Ling-3.1-flash).

## This quant

The file is an EXL3 trellis quantization at 2.10 bpw (153.6 GB). It runs on one 32 GB GPU by keeping the routed experts mostly in system RAM and computing them on the CPU. Calibration was weighted toward agentic coding and tool use, with a general-text floor; it uses exllamav3's standard 2.10 bpw module allocation (`-b 2.0 -hq 2`), so the intended difference from other files at this size is the calibration content.

## Speed, one R9700

Raw continuation, greedy, 256 generated tokens, no speculation and no prefix reuse. Prefill is prompt tokens divided by time to first token; decode excludes the first token. Two measured repetitions after an excluded warm-up.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 377.7 | 11.98 | 21.70 |
| 8K | code | 387.5 | 11.95 | 21.10 |
| 64K | prose | 383.1 | 7.83 | 171.10 |
| 64K | code | 388.7 | 8.15 | 168.60 |
<!-- BENCH:END -->

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 32 GB (gfx1201); AMD Ryzen 7 9800X3D; 192 GB DDR5 |
| Software | Ubuntu 24.04.5 LTS; kernel 7.0.0-28-generic; ROCm 7.2.1; PyTorch 2.9.1+rocm7.2.1.gitff65f5bc; exllamav3 bd4da327a820e3af602b9131c54a41a5736c33ae (+local diff sha256 f2c2c3fd0e625029c22a1bae201d4f2a8ae43b76e53804413f7592803b208547) |
| Run | CPU-MoE layers / threads / chunk 60 (8K) or 62 (64K) / 8 / 4,096; env `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib`, `OMP_NUM_THREADS=8`, `PYTORCH_ALLOC_CONF=expandable_segments:True`; peak VRAM / RAM 31.1 GB / 128.6 GB |
<!-- RECEIPT:END -->

## Quality

Next-token distribution against the FP8 base model on 92 x 4K windows. Lower KL and higher top-1 are better.

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Top-1 |
|---|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.922 | 77.99 % |
| Best other public quant at this size (as of 2026-10-18) | 153.6 GB | 0.962 | 77.66 % |

**Verdict.** Mixed result; neither file is better overall. By KL, Wrench is 4.1% closer to the FP8 model (95% CI -0.052 to -0.027), all of it on the agentic-coding windows; on general text the other quant is slightly closer. The other quant also gives the actual next token higher probability on six full-vocabulary windows (NLL 0.085 lower, 6/6). Deterministic scorer (identical repeats). KL is not task accuracy. Same-config repeat floor: 0.0000.
<!-- QUALITY:END -->

## Runtime techniques

- [exllamav3](https://github.com/turboderp-org/exllamav3) at `0f8cf2c3`, which includes Ling support and the RDNA K = 2 trellis fix ([#519](https://github.com/turboderp-org/exllamav3/pull/519)).
- Routed experts are split between VRAM and system RAM. `-mcl` sets how many expert layers the CPU worker runs; `-mct` sets its thread count.
- Prompt chunk size (`-gcs`) and thread count (`-mct`) were tuned on this box; the benchmark command and receipt record the values used.
- MTP is present in the file but disabled for these measurements.

## Run it

```sh
git clone https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git checkout 0f8cf2c3
git apply /path/to/ling-wrench/bench/exllamav3-rocm-smem.patch   # ROCm, see the known issue below
python -m venv .venv && . .venv/bin/activate
# install the ROCm PyTorch build for your driver first
pip install -r requirements.txt
```

```sh
hf download boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw \
  --local-dir ./Ling-3.1-flash-exl3-wrench-2.10bpw
```

```sh
CUDA_VISIBLE_DEVICES=0 python examples/chat.py \
  -m ./Ling-3.1-flash-exl3-wrench-2.10bpw \
  -mode ling -no_think -temp 0 \
  -mcl 60 -mct 8 \
  -gcs 4096 -cs 66048
```

**Known issue (2026-10-18):** with PyTorch 2.9.1 on ROCm, exllamav3 `0f8cf2c3` (and current upstream) stops at the first decode with `AttributeError: ... has no attribute 'shared_memory_per_block'`. Apply [`bench/exllamav3-rocm-smem.patch`](../bench/exllamav3-rocm-smem.patch) after the checkout, as above. It only changes how the GPU shared-memory limit is looked up (64 KB fallback on ROCm); the numbers on this page were measured with it.

`-mcl` must be high enough that the KV/recurrent cache fits in VRAM at the target context: 60 was measured at 8K and 62 at 64K on a 32 GB card. `-gcs 8192` gave non-finite logits on this card; use 4096.

## Reproduce the benchmark

The inputs are one PG-19 test book (DeepMind, Apache-2.0) and llama.cpp `src/*.cpp` (MIT), tokenized with the Ling tokenizer. Token ids are not redistributed; only the builder and checksums are published.

```sh
python bench/build_inputs.py \
  --tokenizer ./Ling-3.1-flash-exl3-wrench-2.10bpw \
  --out bench/inputs
```

```sh
CUDA_VISIBLE_DEVICES=0 python bench/run_bench.py \
  --model ./Ling-3.1-flash-exl3-wrench-2.10bpw \
  --inputs bench/inputs \
  --exllamav3 ../exllamav3 \
  --cpu-moe-layers 60 --threads 8 --chunk 4096 \
  --reps 2 --warmup 1 \
  --out ling-3.1/results/results.json
```

```sh
python tools/fill_results.py --profile 3.1
python tools/fill_results.py --profile 3.1 --check
```

The public runner is being checked end to end today (it takes one `--cpu-moe-layers` value; the 64K cells used 62). `ling-3.1/results/results.json` records the command, environment, input hashes, timings and peak memory. `bench/README.md` has the full flag list and input-selection rules.

## Credits and license

- Base model: [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), MIT. The upstream copyright and permission notice is reproduced in [NOTICE](../NOTICE).
- Converter and runtime: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT, including the Ling support and the AMD K = 2 trellis fix.
- Benchmark input sources: PG-19 (DeepMind, Apache-2.0) and llama.cpp (MIT).
- The scripts and documentation here are MIT; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). The weights are a derivative of Ling-3.1-flash under the same MIT terms.
