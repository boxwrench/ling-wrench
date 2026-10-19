# Benchmark harness

One harness directory for both releases. Follow [INSTALL](../docs/INSTALL.md) to prepare the runtime and
[REPRODUCE](../docs/REPRODUCE.md) for pins, checksums, commands, timing definitions and limitations.

| Script | Job |
|---|---|
| [build_inputs.py](build_inputs.py) | build 3.1 PG-19/code token windows; writes checksums and a local manifest |
| [run_bench.py](run_bench.py) | 3.1 fresh-generator matrix; one process per context; actual/cached token and output-id receipts |
| [check_determinism.py](check_determinism.py) | one-load 8K-code repeatability screen; does not establish portable output identity |
| [serve_check.py](serve_check.py) | 3.0 API sanity checks plus streamed timings; raw per-server output |
| [exllamav3-rocm-smem.patch](exllamav3-rocm-smem.patch) | shared-memory lookup fallback for the pinned ROCm stack; tested on gfx1201 only |
| [exllamav3-explicit-device.patch](exllamav3-explicit-device.patch) | opt-in `-dev/--device` for `model_init` entry points such as `examples/chat.py`; local, not upstream |
| [serve_fast.py](serve_fast.py) | local OpenAI-compatible chat endpoint with the measured fast loader; stdlib only, one request at a time |

## Runner settings

| Flag | Meaning |
|---|---|
| `--model`, `--inputs`, `--exllamav3` | local weights, generated windows, runtime checkout |
| `--contexts` | comma-separated prompt lengths |
| `--cpu-moe-layers` | one layer count or one per context |
| `--threads`, `--chunk` | CPU expert workers and maximum prompt chunk |
| `--gen-tokens`, `--reps`, `--warmup` | output count, measured repetitions, excluded warm-up passes |
| `--revision`, `--out` | weight revision receipt and output JSON |

Run the commands in REPRODUCE from the prepared venv. The runner rejects nonzero cached tokens/pages,
wrong output counts and non-finite logits. A fresh generator alone is not proof of zero reuse.
Chunk 8192 failed on the measured stack; 4096 is the measured setting.

## Greedy repeatability screen

The one-load screen compared two default and two controlled 256-token runs on R9700,
exllamav3 `0f8cf2c3` plus the patch, ROCm 7.2.1. Both pairs matched within that load;
default pairs elsewhere differed. The controlled arm cost 39.95% prefill time against a 0.10% control prefill-speed spread.
It is not the speed default. The controls are hot-switched in one load, not confirmed in separate fresh processes.

## Results

[3.1 speed/quality sources](../ling-3.1/results/README.md) and
[3.0 combined serving receipt](../ling-3.0/results/runtime.json) feed the release tables.
The current 3.1 JSON is the quiet rerun, not the earlier transfer-overlapped matrix.
The 3.0 API harness output needs manual assembly into the release schema; preserve raw files and failures.
[Community report format](../docs/COMMUNITY_BENCHMARKS.md).

## Explicit single-device loading

`run_bench.py --device cuda:0` selects exllamav3's explicit single-device loader (the visible GPU; with
`CUDA_VISIBLE_DEVICES=1` that is the R9700 on the reference box). Omit `--device` to keep autosplit. The option is
passed to each context subprocess and recorded as `config.load_device`. Explicit loading skips autosplit's VRAM budget
checks, so use a layer count and chunk size measured to fit.

`exllamav3-explicit-device.patch` exposes the same choice to `examples/chat.py` and other `model_init` entry points as
`-dev/--device`; it defaults to autosplit and refuses to combine with `-gs` or `-tp`.

## Fast local server

`serve_fast.py` uses the measured loader settings and serves `POST /v1/chat/completions` on `127.0.0.1` only. It is
non-streaming, handles one request at a time, uses the Ling chat template with thinking off unless the request sets
`"thinking": true`, and is greedy when `temperature` is 0 or absent. **After starting it, send one short request before the first long prompt**: a cold server whose first request was 8K returned `NonFiniteLogits` at step 0; a short request first avoided it (see [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md)). The response's `timings` object carries the
generator's own prefill/decode times and token counts.

```sh
EXL3_MOE_STREAM_T=6 python bench/serve_fast.py -m "$MODEL" \
  -dev cuda:0 -mcl 62 -mct 8 -gcs 8192 -cs 9216 --port 8000
```
