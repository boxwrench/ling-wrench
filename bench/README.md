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
