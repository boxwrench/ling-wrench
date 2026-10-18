# Ling-3.1-flash — Wrench EXL3 2.10 bpw

An EXL3 trellis quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) at 2.10 bits per weight, 153.6 GB. The weights are at [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw); this page holds the measured numbers, the exact run command and the benchmark harness shared with the Ling-3.0 releases.

Part of [boxwrench/ling-wrench](https://github.com/boxwrench/ling-wrench).

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
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.925 | 77.99 % |
| turboderp EXL3 2.10 bpw | 153.6 GB | 0.961 | 77.70 % |
| AtomicChat AD-IQ2_XXS | 174.1 GB | 1.071 | 75.68 % |

**Provisional.** Single runs; KL is not task accuracy. The same-config repeat floor is still being measured.
<!-- QUALITY:END -->

## Runtime techniques

- [exllamav3](https://github.com/turboderp-org/exllamav3) at `0f8cf2c3`, which includes Ling support and the RDNA K = 2 trellis fix ([#519](https://github.com/turboderp-org/exllamav3/pull/519)), plus a one-file ROCm shared-memory fallback (`bench/exllamav3-rocm-smem.patch`).
- Routed experts are split between VRAM and system RAM. `-mcl` sets how many expert layers the CPU worker runs; `-mct` sets its thread count.
- Prompt chunk size (`-gcs`) and thread count (`-mct`) were tuned on this box; the benchmark command and receipt record the values used.
- MTP is present in the file but disabled for these measurements.

## Run it

```sh
git clone https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git checkout 0f8cf2c3
# The ROCm 7.2 "lw" torch wheel omits shared_memory_per_block, which the MLA
# decode path reads. Apply the one-file fallback published next to this page:
curl -LO https://raw.githubusercontent.com/boxwrench/ling-wrench/main/bench/exllamav3-rocm-smem.patch
git apply exllamav3-rocm-smem.patch
python -m venv .venv && . .venv/bin/activate
# The measured build is PyTorch 2.9.1+rocm7.2.1. Install that ROCm wheel for
# your driver; a different torch build will change the numbers.
pip install -r requirements.txt
# On ROCm 7.2 with GCC 13 the extension's AVX2 C++ file cannot compile the HIP
# bf16 headers. Build it with the ROCm clang (the first import also JIT-builds):
export CC=/opt/rocm-7.2.1/llvm/bin/clang CXX=/opt/rocm-7.2.1/llvm/bin/clang++
pip install -e .
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

`-mcl 60` is the measured 8K placement; use `-mcl 62` at 64K (`-gcs 4096` in both cases). `-mcl` must be high enough that the KV/recurrent cache fits in VRAM at the target context; 62 is the minimum safe count at 64K on the R9700. Do not raise `-gcs` to 8192: on this build it produces non-finite logits at step 0.

## Reproduce the benchmark

The inputs are the PG-19 test file (`emozilla/pg19` revision
`c021754c8e01c5b1cc83a1f549c1f97fbbb756b8`, `data/test-00000-of-00001-29a571947c0b5ccc.parquet`;
DeepMind, Apache-2.0) and llama.cpp `src/*.cpp` at
`e467e5b5e5ba43071d6151b3d881deb9300d000e` (MIT), tokenized with the Ling
tokenizer. Token ids are not redistributed; only the builder and checksums are
published.

```sh
pip install pyarrow
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout e467e5b5e5ba43071d6151b3d881deb9300d000e
python bench/build_inputs.py \
  --tokenizer ./Ling-3.1-flash-exl3-wrench-2.10bpw \
  --pg19 /path/to/data/test-00000-of-00001-29a571947c0b5ccc.parquet \
  --code-dir ./llama.cpp \
  --out bench/inputs
```

The four `sha256` values must match `bench/inputs/SHA256SUMS`.

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python bench/run_bench.py \
  --model ./Ling-3.1-flash-exl3-wrench-2.10bpw \
  --inputs bench/inputs \
  --exllamav3 ../exllamav3 \
  --contexts 8192,65536 --cpu-moe-layers 60,62 \
  --threads 8 --chunk 4096 \
  --reps 2 --warmup 1 \
  --out ling-3.1/results/results.json
```

This is the exact configuration and stack in the receipt: exllamav3
`0f8cf2c3` plus the one-file ROCm shared-memory patch
(`bench/exllamav3-rocm-smem.patch`, sha256
`9744c8e95882f25727d9921346611d37f803297eccbdc0d72241f098fa39c0f2`); without it
the first decode raises `AttributeError: shared_memory_per_block`.
`CUDA_VISIBLE_DEVICES=0` assumes one visible GPU; on the reference box the
R9700 is HIP device 1 because a 7900 XT is device 0.

```sh
python tools/fill_results.py --profile 3.1
python tools/fill_results.py --profile 3.1 --check
```

`ling-3.1/results/results.json` records the command, environment, input hashes, timings and peak memory. `bench/README.md` has the full flag list and input-selection rules.

## Credits and license

- Base model: [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), MIT. The upstream copyright and permission notice is reproduced in [NOTICE](../NOTICE).
- Converter and runtime: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT, including the Ling support and the AMD K = 2 trellis fix.
- Benchmark input sources: PG-19 (DeepMind, Apache-2.0) and llama.cpp (MIT).
- The scripts and documentation here are MIT; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). The weights are a derivative of Ling-3.1-flash under the same MIT terms.
