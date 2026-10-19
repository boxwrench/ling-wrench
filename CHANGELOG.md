# v0.4 — 2026-10-19

- Showcases: two one-shot three.js demos by the 3.1 quant with thinking on (`showcase/`), with prompts, reasoning,
  run records and screenshots; live pages via GitHub Pages.
- bartowski IQ2_XXS added to the 3.1 quality table.
- Explicit loader needs an idle card: VRAM headroom notes in INSTALL and TROUBLESHOOTING.

## Fast interactive path (2026-10-18)

- `bench/exllamav3-explicit-device.patch`: opt-in `-dev/--device` for `model_init` entry points such as
  `examples/chat.py`; default remains autosplit, and `-gs`/`-tp` are refused. Local, not upstream.
- `examples/chat.py`-equivalent load with `-dev cuda:0 -nw -mcl 62 -mct 8 -gcs 8192 -cs 9216` and
  `EXL3_MOE_STREAM_T=6`: 502.4 tok/s prefill / 12.24 decode on the public 8K code input; runner 505.8/12.20.
  Without `-nw`, repetitions alternated ~475/~500. The console loop itself was not driven.
- `bench/serve_fast.py`: local OpenAI-compatible chat endpoint with the same loader; stdlib only, one request at a
  time, non-streaming, Ling template, thinking per request, greedy by default.

# Documentation update — 2026-10-18

- Added Linux installation, agent setup, troubleshooting, reproduction and community-report guides.
- Kept concise model pages and linked the shared guides; clarified measured and untested paths.
- Added a relative-link check and single-checkpoint INT4 card rendering; corrected memory-unit labels to GiB.

# v0.3 — 2026-10-18

- Ling-3.1 prefill +38-43%: explicit single-device load (`bench/run_bench.py --device cuda:0`), stream threshold
  `EXL3_MOE_STREAM_T=6`, chunk 8192, 62 CPU-MoE layers. 8K 486.6/498.7, 64K 476.5/476.4 tok/s prefill; decode unchanged
  (12.02/12.07, 7.86/7.64). Six-window exact comparison with FP8: inside the repeat noise.
- `examples/chat.py` still loads with autosplit (~350 tok/s prefill); a faster interactive path is being tested.
- Setup, install, troubleshooting, reproduction and community-benchmark guides; `AGENTS.md`, `llms.txt`.

# v0.2 — 2026-10-18

- Ling-3.1 speed table re-measured on a clean stack (upstream exllamav3 `0f8cf2c3` + the ROCm patch) with the public
  harness on a quiet machine: prefill 346-351 tok/s, decode 12.2-12.3 (8K) / 7.7 (64K). The 2026-10-17 prefill
  (378-389, an older local build) is not reproduced and is withdrawn; decode matches.
- Public input builder reproduces the original input hashes (`bench/inputs/SHA256SUMS` filled); per-context CPU layers in
  `bench/run_bench.py`; exact AMD wheels, compiler and environment in INSTALL; `bench/check_determinism.py`.

# v0.1.1 — 2026-10-18

- Known issue: exllamav3 on ROCm with PyTorch 2.9.1 raises `AttributeError` (`shared_memory_per_block`) at the first decode.
  Added `bench/exllamav3-rocm-smem.patch` and the step to apply it.

# v0.1 — 2026-10-18 (early public release)

- Ling-3.1: R9700 speed table filled (measured 2026-10-17, provisional until today's clean-stack re-run).
- Ling-3.1 quality: identical repeated quality measurements; compared with the best other public quant at this size.
- Ling-3.0: serving receipt lists the full vLLM server flags.
- Run commands use chunk 4096 (the measured setting).

# v0 — 2026-10-17

- Unified the Ling-3.1 EXL3 and Ling-3.0 INT4 release pages in one repository.
- `ling-3.1/`: EXL3 2.10 bpw page, benchmark harness, results placeholders and
  quality JSON.
- `ling-3.0/`: the Wrench and generic INT4 pages, quality table and the vLLM
  serving check with its runtime JSON.
- One `LICENSE` and `NOTICE` covering both inclusionAI base models, exllamav3
  and the benchmark input sources.
- `tools/fill_results.py` renders both the 3.1 benchmark and the 3.0 serving
  results (schema `ling-r9700-bench/1`); its test covers both profiles.

The initial release used placeholders; current measurements are linked from the model pages.
