# Ling-3.1-flash — Wrench EXL3 2.10 bpw

An EXL3 trellis quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) at 2.10 bits per weight, 153.6 GB. The weights are at [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw); this page holds the measured numbers, the exact run command and the benchmark harness shared with the Ling-3.0 releases.

Part of [boxwrench/ling-wrench](https://github.com/boxwrench/ling-wrench).

> **Status, 2026-10-18: verified on a clean stack.** Measured with `bench/run_bench.py` on upstream exllamav3 `0f8cf2c3`
> plus the ROCm shared-memory patch, on a quiet machine. All four input hashes match the original benchmark; all eight
> measured passes generated 256 tokens with zero cached tokens. Decode matches
> the first measurement (2026-10-17, an older local build); its 378-389 tok/s prefill is not reproduced here and is
> withdrawn. Prefill speed-ups are being tested next.

## Base model

Ling-3.1-flash is inclusionAI's 560B-total, ~25B-active hybrid MoE. It has 63 KDA linear-attention layers and 9 gated MLA layers, an MTP head, and a 1M-token context window. It is MIT-licensed; see the [model card](https://huggingface.co/inclusionAI/Ling-3.1-flash).

## This quant

The file is an EXL3 trellis quantization at 2.10 bpw (153.6 GB). It runs on one 32 GB GPU by keeping the routed experts mostly in system RAM and computing them on the CPU. Calibration was weighted toward agentic coding and tool use, with a general-text floor; it uses exllamav3's standard 2.10 bpw module allocation (`-b 2.0 -hq 2`), so the intended difference from other files at this size is the calibration content.

## Speed, one R9700

Reference machine: AMD Radeon AI PRO R9700 32 GB (gfx1201), Ryzen 7 9800X3D
8-core CPU, and 192 GB DDR5. The 7900 XT is unused for compute.

Raw continuation, greedy, 256 generated tokens, no speculation and no prefix reuse. Prefill is prompt tokens divided by time to first token; decode excludes the first token. Two measured repetitions after an excluded warm-up.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | 351.2 | 12.34 | 23.33 |
| 8K | code | 347.6 | 12.21 | 23.57 |
| 64K | prose | 345.8 | 7.72 | 189.51 |
| 64K | code | 348.0 | 7.67 | 188.31 |
<!-- BENCH:END -->

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 32 GB (gfx1201); AMD Ryzen 7 9800X3D; 192 GB DDR5 |
| Software | Ubuntu 24.04.5 LTS; kernel 7.0.0-28-generic; ROCm 7.2.1 (HIP 7.2.53211); PyTorch 2.9.1+rocm7.2.1.gitff65f5bc; exllamav3 0f8cf2c3f052a2503e98c1e77d40b2f055047bcc (+local diff sha256 e9a50674a7ef9a1f728a75cdc7311ae8a65f954887a5109c3aaa55b4787dc439) |
| Run | CPU-MoE layers / threads / chunk 60 (8K) or 62 (64K) / 8 / 4,096; env `CUDA_VISIBLE_DEVICES=1`, `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib`, `OMP_NUM_THREADS=8`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `PYTORCH_ROCM_ARCH=gfx1201`; peak VRAM / RAM 30.9 GB / 133.1 GB |
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

- [exllamav3](https://github.com/turboderp-org/exllamav3) at `0f8cf2c3`, which includes Ling support and the RDNA K = 2 trellis fix ([#519](https://github.com/turboderp-org/exllamav3/pull/519)), plus a one-file ROCm shared-memory fallback (`bench/exllamav3-rocm-smem.patch`).
- Routed experts are split between VRAM and system RAM. `-mcl` sets how many expert layers the CPU worker runs; `-mct` sets its thread count.
- Prompt chunk size (`-gcs`) and thread count (`-mct`) were tuned on this box; the benchmark command and receipt record the values used.
- MTP is present in the file but disabled for these measurements.

## Run it

Use Python 3.12 and the ROCm 7.2.1 development tools. These are the exact
AMD wheels used below; a generic `pip install torch` selects a different stack.
The download is about 1.9 GB. Keep the already-downloaded wheel files if available.

```sh
git clone https://github.com/boxwrench/ling-wrench.git
git clone https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git checkout 0f8cf2c3
git apply ../ling-wrench/bench/exllamav3-rocm-smem.patch
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r requirements.txt -r requirements_examples.txt
export CC=/opt/rocm-7.2.1/llvm/bin/clang
export CXX=/opt/rocm-7.2.1/llvm/bin/clang++
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1   # R9700 on the reference box; use 0 for one GPU
python -m pip install -e .
python -c 'import exllamav3'    # builds/loads the extension; wait for exit 0
export MODEL_DIR=/path/to/Ling-3.1-flash-exl3-wrench-2.10bpw
python examples/chat.py \
  -m "$MODEL_DIR" \
  -mode ling -no_think -temp 0 \
  -mcl 60 -mct 8 -gcs 4096 -cs 9216
```

**Known issue (2026-10-18):** with PyTorch 2.9.1 on ROCm, exllamav3 `0f8cf2c3` (and current upstream) stops at the first decode with `AttributeError: ... has no attribute 'shared_memory_per_block'`. Apply [`bench/exllamav3-rocm-smem.patch`](../bench/exllamav3-rocm-smem.patch) after the checkout, as above. It only changes how the GPU shared-memory limit is looked up (64 KB fallback on ROCm); the numbers on this page were measured with it.

This is the 8K placement. For 64K use `-mcl 62 -cs 66560`, keeping `-mct 8
-gcs 4096`. Higher `-mcl` puts more expert layers in system RAM, leaving room for
the attention cache. Chunk 8192 produced non-finite logits at step 0 on this
stack; use 4096. The shared-memory patch is required on this torch build;
[PyTorch 2.9.1's ROCm source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/cuda/Module.cpp)
also excludes the property, so installing its standard ROCm 6.4 wheel is not a
verified substitute. The 64 KiB fallback is verified here only on the R9700.


## Reproduce the benchmark

The inputs are the PG-19 test file (`emozilla/pg19` revision
`c021754c8e01c5b1cc83a1f549c1f97fbbb756b8`, `data/test-00000-of-00001-29a571947c0b5ccc.parquet`;
DeepMind, Apache-2.0) and llama.cpp `src/*.cpp` at
`e467e5b5e5ba43071d6151b3d881deb9300d000e` (MIT), tokenized with the Ling
tokenizer. Input token ids are not redistributed; only the builder and checksums are
published.

```sh
# Keep the exllamav3 venv and environment above active.
export EXLLAMAV3="$PWD"
cd ../ling-wrench
pip install pyarrow huggingface_hub
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout e467e5b5e5ba43071d6151b3d881deb9300d000e
cp bench/inputs/SHA256SUMS /tmp/ling31-expected.sha256
python bench/build_inputs.py \
  --tokenizer "$MODEL_DIR" \
  --pg19 hf://emozilla/pg19@test \
  --pg19-revision c021754c8e01c5b1cc83a1f549c1f97fbbb756b8 \
  --code-dir ./llama.cpp \
  --out bench/inputs
(cd bench/inputs && sha256sum -c /tmp/ling31-expected.sha256)
```

All four checks must print `OK`. The expected hashes are copied before the
builder overwrites `SHA256SUMS`. A local
copy of the pinned PG-19 test parquet can replace `hf://emozilla/pg19@test`.

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH EXL3_MOE_STREAM_T EXL3_MOE_STREAM_FUSED_T EXL3_MOE_RECON_DET
CUDA_VISIBLE_DEVICES=1 python bench/run_bench.py \
  --model "$MODEL_DIR" \
  --inputs bench/inputs \
  --exllamav3 "$EXLLAMAV3" \
  --contexts 8192,65536 --cpu-moe-layers 60,62 \
  --threads 8 --chunk 4096 \
  --reps 2 --warmup 1 \
  --revision 008db908a26ac55a2999b29617297287b15d187e \
  --out ling-3.1/results/results.json
```

This is the exact configuration and stack in the receipt: exllamav3
`0f8cf2c3` plus the one-file ROCm shared-memory patch
(`bench/exllamav3-rocm-smem.patch`, sha256
`9744c8e95882f25727d9921346611d37f803297eccbdc0d72241f098fa39c0f2`); without it
the first decode raises `AttributeError: shared_memory_per_block`.
`CUDA_VISIBLE_DEVICES=1` selects the R9700 on the reference box; use 0 on a
single-GPU machine. The 7900 XT stays unused.

```sh
python tools/fill_results.py --profile 3.1
python tools/fill_results.py --profile 3.1 --check
```

The runner uses one model process per context, with 60 CPU-MoE layers at 8K and 62 at 64K. `ling-3.1/results/results.json` records the command, environment, input hashes, timings and peak memory. `bench/README.md` has the full flag list and input-selection rules.

The [8K-code repeatability screen](../bench/README.md#greedy-repeatability-screen)
found identical output pairs with and without the deterministic controls in
one load. Those controls cost 39.95% prefill time here and are not enabled in
the speed command. Output identity is not promised across default runs.

## Credits and license

- Base model: [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), MIT. The upstream copyright and permission notice is reproduced in [NOTICE](../NOTICE).
- Converter and runtime: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT, including the Ling support and the AMD K = 2 trellis fix.
- Benchmark input sources: PG-19 (DeepMind, Apache-2.0) and llama.cpp (MIT).
- The scripts and documentation here are MIT; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). The weights are a derivative of Ling-3.1-flash under the same MIT terms.
