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

Benchmark speed numbers and the 3.1 repeatability verdict are still pending; the
pages keep explicit `TBD` placeholders until those measurements land.
