# Ling-3.1-flash — Wrench EXL3 2.10 bpw

An EXL3 trellis quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) at 2.10 bits per weight, 153.6 GB. The weights are at [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw); this repository holds the benchmark harness, the measured numbers and the exact run command.

## Base model

Ling-3.1-flash is inclusionAI's 560B-total, ~25B-active hybrid MoE. It has 63 KDA linear-attention layers and 9 gated MLA layers, an MTP head, and a 1M-token context window. It is MIT-licensed; see the [model card](https://huggingface.co/inclusionAI/Ling-3.1-flash).

## This quant

The file is an EXL3 trellis quantization at 2.10 bpw (153.6 GB). It runs on one 32 GB GPU by keeping the routed experts mostly in system RAM and computing them on the CPU. Calibration was weighted toward agentic coding and tool use, with a general-text floor; the module allocation matches turboderp's 2.10 bpw release, so the intended difference is the calibration content.

## Speed, one R9700

Raw continuation, greedy, 256 generated tokens, no speculation and no prefix reuse. Prefill is prompt tokens divided by time to first token; decode excludes the first token. Two measured repetitions after an excluded warm-up.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | TBD | TBD | TBD |
| 8K | code | TBD | TBD | TBD |
| 64K | prose | TBD | TBD | TBD |
| 64K | code | TBD | TBD | TBD |
<!-- BENCH:END -->

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 32 GB (gfx1201); AMD Ryzen 7 9800X3D; 192 GB DDR5 |
| Software | TBD; kernel TBD; ROCm TBD; PyTorch TBD; exllamav3 TBD |
| Run | CPU-MoE layers / threads / chunk TBD / TBD / TBD; env none; peak VRAM / RAM TBD / TBD |
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

- [exllamav3](https://github.com/turboderp-org/exllamav3) at `0f8cf2c3`, which includes Ling support and the RDNA K = 2 trellis fix ([#519](https://github.com/turboderp-org/exllamav3/pull/519)).
- Routed experts are split between VRAM and system RAM. `-mcl` sets how many expert layers the CPU worker runs; `-mct` sets its thread count.
- Prompt chunk size (`-gcs`) and thread count (`-mct`) were tuned on this box; the benchmark command and receipt record the values used.
- MTP is present in the file but disabled for these measurements.

## Run it

```sh
git clone https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git checkout 0f8cf2c3
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
  -gcs 2048 -cs 66048
```

`-mcl` must be high enough that the KV/recurrent cache fits in VRAM at the target context. This is the starting placement; the receipt records the values kept by the benchmark.

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
  --cpu-moe-layers 60 --threads 8 --chunk 2048 \
  --reps 2 --warmup 1 \
  --out results/results.json
```

```sh
python tools/fill_results.py
python tools/fill_results.py --check
```

`results/results.json` records the command, environment, input hashes, timings and peak memory. `bench/README.md` has the full flag list and input-selection rules.

## Credits and license

- Base model: [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), MIT. The upstream copyright and permission notice is reproduced in [NOTICE](NOTICE).
- Converter and runtime: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT, including the Ling support and the AMD K = 2 trellis fix.
- Benchmark input sources: PG-19 (DeepMind, Apache-2.0) and llama.cpp (MIT).
- The scripts and documentation here are MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The weights are a derivative of Ling-3.1-flash under the same MIT terms.
