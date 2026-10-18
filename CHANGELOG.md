# Documentation update — 2026-10-18

- Added Linux installation, agent setup, troubleshooting, reproduction and community-report guides.
- Kept concise model pages and linked the shared guides; clarified measured and untested paths.
- Added a relative-link check and single-checkpoint INT4 card rendering; corrected memory-unit labels to GiB.

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
