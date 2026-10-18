# Benchmark harness

The harness covers both releases and lives here once.

- `build_inputs.py` builds the public token windows for the Ling-3.1 EXL3 run.
- `run_bench.py` runs the Ling-3.1 four-cell matrix and writes
  `ling-3.1/results/results.json`.
- `serve_check.py` runs the Ling-3.0 serving check against an OpenAI-compatible
  endpoint and writes a runtime JSON.

`run_bench.py` produced the 2026-10-18 verification results in `ling-3.1/results/results.json`
on one R9700 (exllamav3 `0f8cf2c3` plus `exllamav3-rocm-smem.patch`, PyTorch
2.9.1+rocm7.2.1). Its protocol and output keys match the release schema.

## Runtime patch

The measured ROCm 7.2.1 PyTorch wheel omits `shared_memory_per_block` from the device
properties. exllamav3 `0f8cf2c3` reads it in the Triton MLA decode path, so the
first decode raises `AttributeError` on an unmodified checkout. The public PyTorch
2.9.1 source also excludes the property under `USE_ROCM`; changing to the
standard ROCm 6.4 wheel is untested and is not a fix for this lookup. Apply the one-file
fallback before building:

```sh
cd /path/to/exllamav3
git apply /path/to/ling-wrench/bench/exllamav3-rocm-smem.patch
```

The patch sha256 is
`9744c8e95882f25727d9921346611d37f803297eccbdc0d72241f098fa39c0f2`; the results
receipt records the exllamav3 local diff (the runner hashes git output with
the final newline stripped). The 64 KiB fallback was tested only on gfx1201. On ROCm 7.2 with GCC 13, also
build with the ROCm clang (`CC=/opt/rocm-7.2.1/llvm/bin/clang
CXX=/opt/rocm-7.2.1/llvm/bin/clang++`): the extension's AVX2 C++ file includes
HIP bf16 headers that GCC 13 cannot compile.

## Inputs

The inputs are public and reproducible:

- **Prose.** The first book in the PG-19 test split that tokenizes to at least
  70,000 Ling-3.1 tokens. One coherent book, not stitched articles. PG-19 is
  Apache-2.0 (DeepMind). The title and URL are written to the manifest and the
  SHA-256 is recorded in the results.
- **Code.** The `src/*.cpp` files of llama.cpp at
  `e467e5b5e5ba43071d6151b3d881deb9300d000e`, sorted by filename, concatenated
  with a `// ===== FILE: <name> =====` banner. llama.cpp is MIT-licensed.

Both are tokenized exactly as the exllamav3 runtime does it: the model BOS token
is prepended and special-token-looking strings are encoded as text
(`encode_special_tokens`). The first 8,192 ids and the first 65,536 ids are
written as little-endian `uint32` files; for these ids that is the same byte
sequence as the int32 files the reference run hashed. The 8K window is a prefix
of the 64K window. This repository publishes only the builder and the checksums;
the input token ids are not redistributed.

Pinned inputs: PG-19 test (`emozilla/pg19` revision
`c021754c8e01c5b1cc83a1f549c1f97fbbb756b8`, file
`data/test-00000-of-00001-29a571947c0b5ccc.parquet`, sha256
`9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc`) and
llama.cpp `e467e5b5e5ba43071d6151b3d881deb9300d000e`. The committed
`bench/inputs/SHA256SUMS` is the result of that build.

The builder needs `tokenizers` and, for a local parquet file, `pyarrow`; the
`hf://` form also needs `huggingface_hub` (the exllamav3 requirements do not
include either parquet dependency):

```sh
pip install pyarrow huggingface_hub
```

```sh
python bench/build_inputs.py \
  --tokenizer "$MODEL_DIR" \
  --pg19 hf://emozilla/pg19@test \
  --pg19-revision c021754c8e01c5b1cc83a1f549c1f97fbbb756b8 \
  --code-dir /path/to/llama.cpp \
  --out bench/inputs
```

Use a local PG-19 parquet file or directory instead of `hf://...` if the dataset
is already downloaded:

```sh
python bench/build_inputs.py --tokenizer "$MODEL_DIR" \
  --pg19 /path/to/pg19/data/test-00000-of-00001.parquet \
  --code-dir /path/to/llama.cpp --out bench/inputs
```

The script prints one line per file:

```
prose-8k  8192   <sha256>
prose-64k 65536  <sha256>
code-8k   8192   <sha256>
code-64k  65536  <sha256>
```

and writes `SHA256SUMS` plus a local `manifest.json`. `manifest.json` records
the book title/URL and the code file list; it may contain local paths, so it is
git-ignored. `SHA256SUMS` is the file to commit.

## Run

```sh
# Use the venv and environment in ling-3.1/README.md's Run it section.
# The reference R9700 is device 1; use 0 on a single-GPU machine.
unset PYTHONPATH EXL3_MOE_STREAM_T EXL3_MOE_STREAM_FUSED_T EXL3_MOE_RECON_DET
CUDA_VISIBLE_DEVICES=1 python bench/run_bench.py \
  --model "$MODEL_DIR" \
  --inputs bench/inputs \
  --exllamav3 /path/to/exllamav3 \
  --contexts 8192,65536 --cpu-moe-layers 60,62 \
  --threads 8 --chunk 4096 \
  --reps 2 --warmup 1 \
  --revision 008db908a26ac55a2999b29617297287b15d187e \
  --out ling-3.1/results/results.json
```

Flags:

| Flag | Meaning |
|---|---|
| `--model` | local EXL3 snapshot |
| `--inputs` | directory from `build_inputs.py` |
| `--exllamav3` | exllamav3 checkout to import (also `EXLLAMAV3_PATH`) |
| `--contexts` | comma-separated contexts (`8192`, `65536`; default both) |
| `--cpu-moe-layers` | routed-expert layers offloaded to the CPU worker (`-mcl`): one value for all contexts, or one per context (`60,62`) |
| `--threads` | CPU worker threads (`-mct`) |
| `--chunk` | maximum prompt chunk (`-gcs`); the measured value is 4096. **Do not use 8192:** it produces non-finite logits at step 0 on this build |
| `--gen-tokens` | generated tokens per cell (default 256) |
| `--reps` | measured repetitions per cell (default 2) |
| `--warmup` | excluded warm-up passes, 0 or 1 (default 1) |
| `--out` | results JSON path |

The runner measures each context in its own process (the CPU-MoE layer
assignment is fixed at model load), runs an excluded 16-token warm-up **per
cell**, then each cell twice. Each repetition is a fresh job and a fresh
generator with the full prompt, `max_history=0` and no recurrent cache, so no
prefix is reused; the runner fails if the engine reports cached tokens or a
generated-token count other than 256. Sampling is argmax. MTP and any draft model
are not loaded.

## Protocol

- Prefill tok/s = prompt tokens / time to first token.
- TTFT = the engine's `time_prefill`, from the first prefill step to the first
  generated token, in seconds; queue time is excluded.
- Decode tok/s = (generated tokens − 1) / time from first to last token, so the
  first token is not counted twice.
- 256 generated tokens per cell.
- Report the mean of the two measured repetitions and keep both values in
  `cells[].reps`.
- Record peak VRAM (allocator, GiB) and host RAM as MemTotal minus minimum MemAvailable (GiB); process RSS
  omits most of the CPU-expert worker tier. The runner's host-memory monitor
  starts after model load; a separate supervisor captured load-time peaks for
  the verification run.

exllamav3 keeps completed cache pages for prefix reuse. The runner disables that
(`max_history=0`) and rejects a repetition whose result reports `cached_tokens >
0`, so every measured pass is the full prompt. The published results state this
protocol. Generated continuation ids and hashes are retained in the results JSON;
input id files remain local.

## Output

`ling-3.1/results/results.json` uses the shared schema:

```json
{"schema":"ling-r9700-bench/1","kind":"exl3-bench","date":"YYYY-MM-DD",
 "model":{"repo":"...","revision":"...","bpw":2.10,"size_gb":153.6},
 "hardware":{"gpu":"...","cpu":"...","ram":"...","os":"...","kernel":"...","rocm":"...","torch":"...","exllamav3":"..."},
 "config":{"cpu_threads":8,"chunk":4096,"cpu_moe_layers":null,
           "cpu_moe_layers_8k":60,"cpu_moe_layers_64k":62,"env":{},"cmd":"..."},
 "inputs":[{"name":"prose-8k","source":"PG-19 ...","tokens":8192,"sha256":"..."}],
 "cells":[{"context":8192,"domain":"prose","prefill_tps":0,"decode_tps":0,"ttft_s":0,
           "gen_tokens":256,"reps":[{"prefill_tps":0,"decode_tps":0,"ttft_s":0}]}],
 "baseline":{"config":{},"cells":[]},
 "screen":[],"peak_vram_gb":0,"peak_ram_gb":0}
```

`tools/fill_results.py --profile 3.1` reads this file and rewrites the marker
blocks in `ling-3.1/README.md`, `ling-3.1/results/README.md` and an optional
Hugging Face card draft.

## Greedy repeatability screen

The four-cell speed table uses the default environment above. To repeat the
8K-code, one-load screen (two 256-token passes per arm, 16-token warm-up per
arm, fresh generators), run:

```sh
unset EXL3_MOE_STREAM_T EXL3_MOE_STREAM_FUSED_T EXL3_MOE_RECON_DET
CUDA_VISIBLE_DEVICES=1 python bench/check_determinism.py \
  --model "$MODEL_DIR" --inputs bench/inputs --exllamav3 "$EXLLAMAV3" \
  --out results/determinism/results.json
sha256sum results/determinism/det-*.bin
```

On this R9700 both pairs were identical, with the same 256 generated ids. Their
little-endian uint32 SHA-256 is
`16c8be2d5422003fb751fd4cac77fffb9f05782c1af8bfe112ecb9659b68365c`.
Default output was not identical in several matrix pairs, so this one pair
does not establish general default determinism.

| Metric | Default | Deterministic | Delta | Default pair spread |
|---|---:|---:|---:|---:|
| Prefill tok/s | 348.95 | 249.34 | -28.55% | 0.10% |
| Decode tok/s | 11.93 | 12.05 | +0.99% | 0.11% |

The deterministic arm applies the upstream controls `EXL3_MOE_STREAM_T=8`,
`EXL3_MOE_STREAM_FUSED_T=0`, `EXL3_MOE_RECON_DET=1` through their corresponding
loaded tuning objects and stream states, allowing both arms to share one
model load. **Not adopted for the speed command:** prefill time increases
39.95%, above the 3% cost limit and the measured floor. This is an 8K-code
screen, not a four-cell deterministic speed claim.

## Ling-3.0 serving check

`serve_check.py` is the one harness for the two INT4 checkpoints. It uses only the
OpenAI-compatible API plus the server's `/tokenize` and `/detokenize` helpers:

```sh
python bench/serve_check.py http://127.0.0.1:8000/v1 auto \
  ling-3.0/results/runtime.json 3
```

It sends four greedy prompts (factual, thinking-on reasoning, a well-formed tool
call, short code), then measures prompt and decode throughput at about 1K and
8K tokens, three runs per point. The prompt is built by tokenizing a public
text, slicing the ids and detokenizing, so the token counts come from the
server. No prefix is reused. Use the same runtime, format and launch settings
for the official INT4 and both re-roundings.

The runtime JSON uses the same `ling-r9700-bench/1` schema with
`kind: serving-check`. `tools/fill_results.py --profile 3.0` renders its
`RUNTIME` and `RECEIPT` blocks.
