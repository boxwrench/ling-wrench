# Ling-3.1-flash — Wrench EXL3 2.10 bpw

A 2.10-bits-per-weight EXL3 trellis quant of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), produced with the upstream exllamav3 converter and a calibration mix weighted toward agentic coding and tool use. The file is 153.6 GB and runs on one 32 GB AMD Radeon AI PRO R9700 by keeping part of the routed-expert stack on the CPU (exllamav3 CPU-MoE offload). It uses the same module-for-module strategy as turboderp's generic 2.10 bpw EXL3 release — `-b 2.0 -hq 2`, head 6 bits, MTP 2 bits, `mul1` codebook, 250 × 2048 calibration tokens — so the only intended difference is the calibration content. This repository publishes the conversion recipe, the public evaluation method, the single-card benchmark harness and the measured numbers. The weights live on Hugging Face at [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw).

## Headline speed, one R9700

Raw continuation, greedy, 256 generated tokens, no speculation, no prefix reuse. Prefill is prompt tokens divided by time to first token; decode excludes the first token. Two measured repetitions after an excluded warm-up.

<!-- BENCH:START -->
| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |
|---:|---|---:|---:|---:|
| 8K | prose | TBD | TBD | TBD |
| 8K | code | TBD | TBD | TBD |
| 64K | prose | TBD | TBD | TBD |
| 64K | code | TBD | TBD | TBD |
<!-- BENCH:END -->

The table is rendered from `results/results.json` by `tools/fill_results.py`. A number is shown only when the benchmark has written it.

## Hardware and software receipt

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 32 GB (gfx1201) |
| CPU | AMD Ryzen 7 9800X3D |
| RAM | 192 GB DDR5 |
| OS | TBD |
| Kernel | TBD |
| ROCm | TBD |
| PyTorch | TBD |
| exllamav3 | TBD |
| Model revision | TBD |
| CPU-MoE layers / threads / chunk | TBD / TBD / TBD |
| Env | none recorded |
| Command | TBD |
| Peak VRAM / RAM | TBD / TBD |
<!-- RECEIPT:END -->

## Quality

Next-token distribution against the FP8 base model on a locked 92 × 4096-token corpus. `KL top-64` and `KL_pred` use the candidate's top-64 distribution over 92 windows; `exact KL` and `exact dNLL` use full-vocabulary scoring on the 6-window prefix. Lower KL is better; lower (more negative) dNLL means the candidate assigns more probability to the actual next token than the FP8 reference does.

<!-- QUALITY:START -->
| Candidate | Size | KL top-64 | Exact KL | KL_pred | Top-1 | Exact dNLL |
|---|---:|---:|---:|---:|---:|---:|
| Wrench EXL3 2.10 bpw | 153.6 GB | 0.925 | 1.161 | 0.130 | 77.99 % | -0.022 |
| turboderp EXL3 2.10 bpw | 153.6 GB | 0.961 | 1.187 | 0.130 | 77.70 % | -0.092 |
| AtomicChat AD-IQ2_XXS | 174.1 GB | 1.071 | 1.330 | 0.176 | 75.68 % | -0.062 |

**Verdict pending a noise-floor check.** The numbers above are single-run rankings; the same-configuration repeat noise floor is still being measured. Treat the ordering as provisional until it is reported.
<!-- QUALITY:END -->

At the same size, the Wrench calibration lowers plain KL overall and on agentic session traffic, and raises top-1 on that half. It is slightly worse on public-domain prose and wikitext, and its exact actual-token dNLL is worse than turboderp's on the 6-window prefix. Both files are far closer to the FP8 reference than the 2-bit IQ2 file on every window-based metric. The method, corpus and uncertainty procedure are in [eval/METHOD.md](eval/METHOD.md); the per-window numbers and bootstrap intervals are in the benchmark report that accompanies a filled `results/results.json`.

## Run it

### 1. Install exllamav3 at a pinned commit

The conversion and benchmark use the ROCm trellis path. PR [#519](https://github.com/turboderp-org/exllamav3/pull/519) makes the K = 2 dense encoder fit in 64 KB of LDS on RDNA; pin a commit that contains it.

```sh
git clone https://github.com/turboderp-org/exllamav3.git
cd exllamav3
git checkout 0f8cf2c3   # K = 2 dense-trellis LDS fix (#519), on top of the Ling support commit
python -m venv .venv
. .venv/bin/activate
# Install the ROCm PyTorch build that matches your driver first, then:
pip install -r requirements.txt
```

On ROCm, build the extensions with `ROCM_HOME` set and the target architecture selected. For a gfx1201 card:

```sh
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1201
```

### 2. Download the model

```sh
export MODEL_DIR="$PWD/models/Ling-3.1-flash-exl3-wrench-2.10bpw"
hf download boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw --local-dir "$MODEL_DIR"
```

### 3. Chat

```sh
CUDA_VISIBLE_DEVICES=0 python examples/chat.py \
  -m "$MODEL_DIR" \
  -mode ling -no_think -temp 0 \
  -mcl 60 -mct 8 \
  -gcs 2048 -cs 66048
```

`-mcl` is the number of routed-expert layers kept in system RAM and run by the CPU worker; it has to be high enough that the KV/recurrent cache for the target context fits in VRAM. `-mct` is the CPU worker thread count. Drop `-mcl` and `-mct` only if the whole file fits on your card, which it does not on a 32 GB device. For a 64K prompt, verify that the cache allocation succeeds before benchmarking; reduce `-mcl` if it does not.

## Reproduce the benchmark

Build the token inputs once. The script takes the first PG-19 test book with at least 70,000 Ling tokens and a concatenation of llama.cpp `src/*.cpp`, tokenizes both with the Ling tokenizer and writes exactly 8,192- and 65,536-token ID files plus SHA-256 hashes.

```sh
python bench/build_inputs.py \
  --tokenizer "$MODEL_DIR" \
  --out bench/inputs
```

Run the four cells on one card:

```sh
CUDA_VISIBLE_DEVICES=0 python bench/run_bench.py \
  --model "$MODEL_DIR" \
  --inputs bench/inputs \
  --cpu-moe-layers 60 --threads 8 --chunk 2048 \
  --reps 2 --warmup 1 \
  --out results/results.json
```

Then render the pages and verify that nothing is left unfilled:

```sh
python tools/fill_results.py
python tools/fill_results.py --check
```

The runner records the exact command, environment, input hashes, per-repetition timings and peak memory in `results/results.json` (schema `ling31-r9700-bench/1`). `bench/README.md` has the full flag list and the input-selection rules.

## Quantization method

The Wrench file is an EXL3 trellis quant built with `-b 2.0 -hq 2 -hb 6 -mb 2 --codebook mul1 --out_scales auto` and 250 sequences of 2,048 tokens. The allocation is the same as turboderp's 2.10 bpw file module for module: 104,448 tensors at 2 bits, 12 at 4 bits, 493 at 6 bits, the rest unquantized; 24 shards; identical tensor names and byte sizes. Only the calibration data differs.

The Wrench mix is:

| Share | Content |
|---:|---|
| 45 % | agentic coding and tool-use sessions from our own workflow, excluded from the evaluation corpus |
| 15 % | public coding and agentic traces |
| 10 % | the model's own thinking traces |
| 30 % | generalist floor: English, Chinese, long documents, math and chat |

The session share is described at this level on purpose; the calibration text itself is not redistributed. The conversion command, the exact exllamav3 commit, the RDNA encoder fix and the measured wall time are in [quant/RECIPE.md](quant/RECIPE.md).

## Caveats

- **One corpus.** Quality is measured on a fixed 92 × 4096-token corpus (agentic sessions, public chat, wikitext). It is not a task-success benchmark.
- **KL is not accuracy.** Lower KL means the next-token distribution is closer to FP8. It does not prove that an agent completes more tasks; only a task harness can show that.
- **Single box.** Speed is one R9700 with CPU-MoE offload on one desktop. A different CPU, RAM speed, driver or power limit will move the numbers, especially prefill.
- **Calibration trade-off.** The mix favours agentic traffic. Public-domain prose and wikitext are slightly worse than turboderp's generic calibration at the same size, and the exact actual-token dNLL on the 6-window prefix does not confirm the private-half KL win.
- **No speculation.** MTP is present in the file but disabled for these measurements; enabling it changes both speed and output.
- **Provisional quality ranking.** The repeat noise floor is still being measured; see the note in the quality block.

## Credits

- Base model: [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash), MIT.
- Converter and runtime: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT, including the Ling support work and the AMD K = 2 trellis fix ([#519](https://github.com/turboderp-org/exllamav3/pull/519)).
- EXL3 / QTIP trellis quantization: the QTIP authors and the exllamav3 contributors.
- Reference 2.10 bpw file used as the matched control: turboderp's EXL3 release.
- Comparison 2-bit file: AtomicChat AD-IQ2_XXS.

## License

The scripts and documentation in this repository are MIT-licensed; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights are a derivative of Ling-3.1-flash and are distributed under the same MIT terms as the base model.
