#!/usr/bin/env python3
"""Tiny test for tools/fill_results.py using a fake results.json.

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
    "schema": "ling31-r9700-bench/1",
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

SKELETON = """# Page

Intro text.

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


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*args, expect=0):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                       text=True, capture_output=True)
    if r.returncode != expect:
        raise AssertionError(f"exit {r.returncode} != {expect}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    return r


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "results").mkdir()
        (root / "README.md").write_text(SKELETON, encoding="utf-8")
        (root / "results" / "README.md").write_text(SKELETON, encoding="utf-8")
        (root / "card.md").write_text(SKELETON, encoding="utf-8")
        rp = root / "results" / "results.json"
        qp = root / "results" / "quality.json"
        rp.write_text(json.dumps(RESULTS), encoding="utf-8")
        qp.write_text(json.dumps(QUALITY), encoding="utf-8")

        run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
        readme = (root / "README.md").read_text(encoding="utf-8")
        assert "TBD" not in readme, readme
        assert "123.4" in readme and "1088.70" in readme, readme
        assert "31.2 GB / 96.5 GB" in readme, readme
        assert "Wrench is ahead" in readme, readme
        assert "0.0020" in readme, readme

        h1 = digest(root / "README.md")
        run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
        assert digest(root / "README.md") == h1, "renderer is not idempotent"

        run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md", "--check")

        # Placeholder JSON -> fill leaves TBD -> check must fail.
        rp.write_text(json.dumps({
            "schema": "ling31-r9700-bench/1",
            "date": None,
            "model": {"repo": "x", "revision": None, "bpw": 2.1, "size_gb": 153.6},
            "hardware": {},
            "config": {},
            "inputs": [],
            "cells": [],
            "baseline": {"config": {}, "cells": []},
            "screen": [],
            "peak_vram_gb": None,
            "peak_ram_gb": None,
        }), encoding="utf-8")
        run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
        r = run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md", "--check",
                expect=1)
        assert "TBD" in r.stderr

        # Missing card is allowed once the in-repo pages are complete.
        rp.write_text(json.dumps(RESULTS), encoding="utf-8")
        run("--root", root, "--results", rp, "--quality", qp, "--card", root / "card.md")
        run("--root", root, "--results", rp, "--quality", qp,
            "--card", root / "does-not-exist.md", "--check")

    print("ok")


if __name__ == "__main__":
    main()
