# Benchmark harness

The harness covers both releases and lives here once.

- `build_inputs.py` builds the public token windows for the Ling-3.1 EXL3 run.
- `run_bench.py` runs the Ling-3.1 four-cell matrix and writes
  `ling-3.1/results/results.json`.
- `serve_check.py` runs the Ling-3.0 serving check against an OpenAI-compatible
  endpoint and writes a runtime JSON.

`run_bench.py` is a draft until the R9700 benchmark run replaces it with the
exact script that produced the published `results.json`. The protocol and output
keys match the release schema; the first real run is the shakedown.

## Inputs

The inputs are public and reproducible:

- **Prose.** The first book in the PG-19 test split that tokenizes to at least
  70,000 Ling-3.1 tokens. One coherent book, not stitched articles. PG-19 is
  Apache-2.0 (DeepMind). The title and URL are written to the manifest and the
  SHA-256 is recorded in the results.
- **Code.** The `src/*.cpp` files of a llama.cpp checkout, sorted by filename,
  concatenated with a one-line file banner. llama.cpp is MIT-licensed.

Both are tokenized with the Ling-3.1 tokenizer (`tokenizer.json` from the model
snapshot) with no added special tokens. The first 8,192 ids and the first 65,536
ids are written as little-endian `uint32` files. The 8K window is a prefix of
the 64K window. This repository publishes only the builder and the checksums;
the token ids are not redistributed.

```sh
python bench/build_inputs.py \
  --tokenizer "$MODEL_DIR" \
  --pg19 hf://emozilla/pg19@test \
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
CUDA_VISIBLE_DEVICES=0 python bench/run_bench.py \
  --model "$MODEL_DIR" \
  --inputs bench/inputs \
  --exllamav3 /path/to/exllamav3 \
  --cpu-moe-layers 60 --threads 8 --chunk 4096 \
  --reps 2 --warmup 1 \
  --out ling-3.1/results/results.json
```

Flags:

| Flag | Meaning |
|---|---|
| `--model` | local EXL3 snapshot |
| `--inputs` | directory from `build_inputs.py` |
| `--exllamav3` | exllamav3 checkout to import (also `EXLLAMAV3_PATH`) |
| `--cpu-moe-layers` | routed-expert layers offloaded to the CPU worker (`-mcl`) |
| `--threads` | CPU worker threads (`-mct`) |
| `--chunk` | maximum prompt chunk (`-gcs`) |
| `--gen-tokens` | generated tokens per cell (default 256) |
| `--reps` | measured repetitions per cell (default 2) |
| `--warmup` | excluded warm-up passes, 0 or 1 (default 1) |
| `--out` | results JSON path |

The runner loads one model, runs an excluded 16-token warm-up on the first cell,
then each cell twice. Each repetition is a fresh job with the full prompt, so the
runner does not reuse a prefix explicitly. Sampling is argmax. MTP and any draft model are not loaded.

## Protocol

- Prefill tok/s = prompt tokens / time to first token.
- TTFT = time from job submission to the first generated token, in seconds.
- Decode tok/s = (generated tokens − 1) / time from first to last token, so the
  first token is not counted twice.
- 256 generated tokens per cell.
- Report the mean of the two measured repetitions and keep both values in
  `cells[].reps`.
- Record peak VRAM (allocator) and peak RSS for the process tree.

exllamav3 keeps completed cache pages for prefix reuse. The draft runner submits
a fresh job for every repetition but does not reset the paged cache; if a strict
no-prefix-reuse measurement is required, run each cell in a fresh process. The
published results must state which of the two was used. The R9700 run replaces
this draft with the exact script.

## Output

`ling-3.1/results/results.json` uses the shared schema:

```json
{"schema":"ling-r9700-bench/1","kind":"exl3-bench","date":"YYYY-MM-DD",
 "model":{"repo":"...","revision":"...","bpw":2.10,"size_gb":153.6},
 "hardware":{"gpu":"...","cpu":"...","ram":"...","os":"...","kernel":"...","rocm":"...","torch":"...","exllamav3":"..."},
 "config":{"cpu_threads":8,"chunk":2048,"cpu_moe_layers":60,"env":{},"cmd":"..."},
 "inputs":[{"name":"prose-8k","source":"PG-19 ...","tokens":8192,"sha256":"..."}],
 "cells":[{"context":8192,"domain":"prose","prefill_tps":0,"decode_tps":0,"ttft_s":0,
           "gen_tokens":256,"reps":[{"prefill_tps":0,"decode_tps":0,"ttft_s":0}]}],
 "baseline":{"config":{},"cells":[]},
 "screen":[],"peak_vram_gb":0,"peak_ram_gb":0}
```

`tools/fill_results.py --profile 3.1` reads this file and rewrites the marker
blocks in `ling-3.1/README.md`, `ling-3.1/results/README.md` and an optional
Hugging Face card draft.

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
