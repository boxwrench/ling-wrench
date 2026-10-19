#!/usr/bin/env python3
"""Test tools/fill_results.py for both release profiles.

Run with:  python tools/test_fill_results.py
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "fill_results.py"

RESULTS = {
    "schema": "ling-r9700-bench/1",
    "kind": "exl3-bench",
    "date": "2026-10-17",
    "model": {"repo": "boxwrench/test", "revision": "abc123", "bpw": 2.1, "size_gb": 153.6},
    "hardware": {
        "gpu": "Test GPU 32 GB", "cpu": "Test CPU", "ram": "192 GB",
        "os": "Linux 6.0", "kernel": "6.0", "rocm": "7.2.1",
        "torch": "2.9.1+rocm7.2.1", "exllamav3": "0f8cf2c3 (no local diff)",
    },
    "config": {
        "cpu_threads": 8, "chunk": 2048, "cpu_moe_layers": 60,
        "env": {"CUDA_VISIBLE_DEVICES": "0"}, "cmd": "run_bench.py --test",
    },
    "inputs": [
        {"name": "prose-8k", "source": "PG-19 test", "tokens": 8192, "sha256": "00"},
        {"name": "prose-64k", "source": "PG-19 test", "tokens": 65536, "sha256": "01"},
        {"name": "code-8k", "source": "llama.cpp", "tokens": 8192, "sha256": "02"},
        {"name": "code-64k", "source": "llama.cpp", "tokens": 65536, "sha256": "03"},
    ],
    "cells": [
        {"context": 8192, "domain": "prose", "prefill_tps": 123.4, "decode_tps": 12.34,
         "ttft_s": 66.39, "gen_tokens": 256, "reps": []},
        {"context": 8192, "domain": "code", "prefill_tps": 130.1, "decode_tps": 12.20,
         "ttft_s": 62.98, "gen_tokens": 256, "reps": []},
        {"context": 65536, "domain": "prose", "prefill_tps": 60.2, "decode_tps": 9.10,
         "ttft_s": 1088.70, "gen_tokens": 256, "reps": []},
        {"context": 65536, "domain": "code", "prefill_tps": 62.0, "decode_tps": 9.00,
         "ttft_s": 1056.71, "gen_tokens": 256, "reps": []},
    ],
    "baseline": {"config": {}, "cells": []},
    "screen": [],
    "peak_vram_gb": 31.2,
    "peak_ram_gb": 96.5,
}

QUALITY = {
    "schema": "ling31-wrench-quality/1",
    "date": "2026-10-17",
    "status": "final",
    "reference": "test FP8",
    "corpus": "92 x 4096",
    "noise_floor": 0.002,
    "verdict": "Wrench is ahead on plain KL and tied on top-1.",
    "candidates": [
        {"name": "Wrench", "size_gb": 153.6, "kl_top64": 0.9247, "exact_kl": 1.1611,
         "kl_pred": 0.130, "top1_pct": 77.99, "dnll_exact": -0.022},
        {"name": "Control", "size_gb": 153.6, "kl_top64": 0.9606, "exact_kl": 1.1866,
         "kl_pred": 0.130, "top1_pct": 77.70, "dnll_exact": -0.092},
    ],
}

SERVING = {
    "schema": "ling-r9700-bench/1",
    "kind": "serving-check",
    "date": "2026-10-17",
    "protocol": "Greedy; 64 generated tokens; three runs per point.",
    "runtime": {"name": "vLLM", "version": "0.26.1", "backend": "ROCm",
                "model_format": "compressed-tensors S4/g32"},
    "hardware": {"gpu": "Test APU 128 GB", "cpu": "Test CPU", "ram": "128 GB unified",
                 "os": "Linux"},
    "models": [
        {"name": "Official INT4", "repo": "inclusionAI/test", "status": "SERVES OK",
         "points": [
             {"target": 1024, "runs": [
                 {"run": 1, "prefill_tps": 958.9, "decode_tps": 27.0, "ttft_s": 1.09},
                 {"run": 2, "prefill_tps": 961.9, "decode_tps": 29.6, "ttft_s": 1.09},
                 {"run": 3, "prefill_tps": 960.1, "decode_tps": 29.6, "ttft_s": 1.09}]},
             {"target": 8192, "runs": [
                 {"run": 1, "prefill_tps": 1081.0, "decode_tps": 25.4, "ttft_s": 7.59},
                 {"run": 2, "prefill_tps": 1082.9, "decode_tps": 28.6, "ttft_s": 7.60},
                 {"run": 3, "prefill_tps": 1081.6, "decode_tps": 26.2, "ttft_s": 7.59}]},
         ]},
        {"name": "Wrench", "repo": "boxwrench/test", "status": "SERVES OK",
         "points": [
             {"target": 1024, "runs": [
                 {"run": 1, "prefill_tps": 949.6, "decode_tps": 29.7, "ttft_s": 1.10},
                 {"run": 2, "prefill_tps": 962.1, "decode_tps": 27.4, "ttft_s": 1.09},
                 {"run": 3, "prefill_tps": 961.4, "decode_tps": 27.9, "ttft_s": 1.09}]},
             {"target": 8192, "runs": [
                 {"run": 1, "prefill_tps": 891.1, "decode_tps": 27.0, "ttft_s": 9.22},
                 {"run": 2, "prefill_tps": 1084.2, "decode_tps": 28.7, "ttft_s": 7.58},
                 {"run": 3, "prefill_tps": 1083.0, "decode_tps": 27.0, "ttft_s": 7.58}]},
         ]},
    ],
}

SKELETON_31 = """# Page

<!-- BENCH:START -->
old bench
<!-- BENCH:END -->

<!-- RECEIPT:START -->
old receipt
<!-- RECEIPT:END -->

<!-- QUALITY:START -->
old quality
<!-- QUALITY:END -->
"""

SKELETON_30 = """# Page

<!-- RUNTIME:START -->
old runtime
<!-- RUNTIME:END -->

<!-- RECEIPT:START -->
old receipt
<!-- RECEIPT:END -->
"""


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*args, expect=0):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                       text=True, capture_output=True)
    if r.returncode != expect:
        raise AssertionError(f"exit {r.returncode} != {expect}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    return r


def test_31(root):
    (root / "results").mkdir(parents=True)
    (root / "README.md").write_text(SKELETON_31, encoding="utf-8")
    (root / "results" / "README.md").write_text(SKELETON_31, encoding="utf-8")
    (root / "card.md").write_text(SKELETON_31, encoding="utf-8")
    rp = root / "results" / "results.json"
    qp = root / "results" / "quality.json"
    rp.write_text(json.dumps(RESULTS), encoding="utf-8")
    qp.write_text(json.dumps(QUALITY), encoding="utf-8")

    run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "TBD" not in readme, readme
    assert "123.4" in readme and "1088.70" in readme, readme
    assert "31.2 GiB / 96.5 GiB" in readme, readme
    assert "Wrench is ahead" in readme, readme
    assert "0.0020" in readme, readme

    h1 = digest(root / "README.md")
    run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
    assert digest(root / "README.md") == h1, "3.1 renderer is not idempotent"
    run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md", "--check")

    # Placeholder JSON -> fill leaves TBD -> check must fail.
    rp.write_text(json.dumps({
        "schema": "ling-r9700-bench/1",
        "kind": "exl3-bench",
        "model": {"repo": "x", "revision": None, "bpw": 2.1, "size_gb": 153.6},
        "hardware": {}, "config": {}, "inputs": [], "cells": [],
        "baseline": {"config": {}, "cells": []}, "screen": [],
        "peak_vram_gb": None, "peak_ram_gb": None,
    }), encoding="utf-8")
    run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
    r = run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md",
            "--check", expect=1)
    assert "TBD" in r.stderr

    # Missing card is allowed once the in-repo pages are complete.
    rp.write_text(json.dumps(RESULTS), encoding="utf-8")
    run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
    run("--root", root, "--results", rp, "--quality", qp,
        "--card", root / "does-not-exist.md", "--check")


def test_30(root):
    root.mkdir(parents=True)
    (root / "README.md").write_text(SKELETON_30, encoding="utf-8")
    rp = root / "runtime.json"
    rp.write_text(json.dumps(SERVING), encoding="utf-8")
    run("--root", root, "--results", rp, "--page", root / "README.md")
    page = (root / "README.md").read_text(encoding="utf-8")
    assert "TBD" not in page, page
    assert "| Official INT4 | SERVES OK | 1K | 960.1 [958.9-961.9] | 29.60 [27.00-29.60] | 1.09 [1.09-1.09] |" in page, page
    assert "| Wrench | SERVES OK | 8K | 1083.0 [891.1-1084.2]" in page, page
    assert "compressed-tensors S4/g32" in page, page
    h1 = digest(root / "README.md")
    run("--root", root, "--results", rp, "--page", root / "README.md")
    assert digest(root / "README.md") == h1, "3.0 renderer is not idempotent"
    run("--root", root, "--results", rp, "--page", root / "README.md", "--check")

    # A single-checkpoint card must not inherit the official or other file's rows.
    card = root / "card.md"
    card.write_text("<!-- MODEL_REPO:boxwrench/test -->\n" + SKELETON_30)
    run("--root", root, "--results", rp, "--page", root / "README.md", "--card", card)
    rendered = card.read_text()
    assert "| Wrench | SERVES OK | 8K | 1083.0" in rendered, rendered
    assert "| Official INT4 |" not in rendered, rendered
    before = digest(card)
    run("--root", root, "--results", rp, "--page", root / "README.md", "--card", card)
    assert digest(card) == before, "3.0 card renderer is not idempotent"

    # A serving JSON with no models leaves TBD -> check must fail.
    rp.write_text(json.dumps({"schema": "ling-r9700-bench/1", "kind": "serving-check",
                              "runtime": {}, "hardware": {}, "models": []}), encoding="utf-8")
    run("--root", root, "--results", rp, "--page", root / "README.md")
    r = run("--root", root, "--results", rp, "--page", root / "README.md", "--check", expect=1)
    assert "TBD" in r.stderr


def test_both(root):
    (root / "ling-3.1" / "results").mkdir(parents=True)
    (root / "ling-3.0" / "results").mkdir(parents=True)
    (root / "ling-3.1" / "README.md").write_text(SKELETON_31, encoding="utf-8")
    (root / "ling-3.1" / "results" / "README.md").write_text(SKELETON_31, encoding="utf-8")
    (root / "ling-3.0" / "README.md").write_text(SKELETON_30, encoding="utf-8")
    (root / "ling-3.1" / "results" / "results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    (root / "ling-3.1" / "results" / "quality.json").write_text(json.dumps(QUALITY), encoding="utf-8")
    (root / "ling-3.0" / "results" / "runtime.json").write_text(json.dumps(SERVING), encoding="utf-8")

    run("--root", root)
    assert "123.4" in (root / "ling-3.1" / "README.md").read_text(encoding="utf-8")
    assert "1083.0" in (root / "ling-3.0" / "README.md").read_text(encoding="utf-8")
    run("--root", root, "--check")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_31(root / "a")
        test_30(root / "b")
        test_both(root / "c")
    print("ok")


if __name__ == "__main__":
    main()
