#!/usr/bin/env python3
"""DRAFT exllamav3 runner for the Ling-3.1 one-R9700 benchmark.

This is the runner shape the release pages consume. It was written on a desk
machine without a GPU and has NOT been exercised end to end against the R9700;
treat the first real run as a shakedown. It implements the protocol in
`bench/README.md`: 8K and 64K context, prose and code, greedy, 256 new tokens,
two measured repetitions after an excluded warm-up, no speculation, no prefix
reuse.

Output conforms to schema `ling31-r9700-bench/1` (the exact keys are listed in
the benchmark brief and in `results/results.json`).
"""

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

DOMAINS = ("prose", "code")
CONTEXTS = (8192, 65536)
DEFAULT_GEN = 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path):
    import torch

    raw = Path(path).read_bytes()
    if len(raw) % 4:
        raise SystemExit(f"{path}: not a uint32 id file")
    n = len(raw) // 4
    ids = list(struct.unpack(f"<{n}I", raw))
    return torch.tensor([ids], dtype=torch.long)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_state(path):
    if not path or not Path(path).exists():
        return None, None
    commit = sh(f"git -C {path} rev-parse --short=12 HEAD")
    diff = sh(f"git -C {path} diff | sha256sum")
    diff = diff.split()[0] if diff else None
    return commit, diff


def peak_rss_gb():
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except Exception:
                pass
        return rss / 1024 ** 3
    except Exception:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024 ** 2
        except Exception:
            pass
        return None


def env_snapshot():
    keys = ("CUDA_VISIBLE_DEVICES", "ROCM_HOME", "PYTORCH_ROCM_ARCH", "HIP_LAUNCH_BLOCKING",
            "EXL3_MOE_CPU_THREADS", "EXL3_MOE_CPU_OFFLOAD", "EXL3_MOE_HOST_CORES",
            "EXL3_MOE_CPU_SWAP", "EXL3_FE_GEMM", "PYTORCH_CUDA_ALLOC_CONF")
    return {k: os.environ[k] for k in keys if k in os.environ}


def load_manifest(inputs):
    p = Path(inputs) / "manifest.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def input_rows(inputs):
    man = load_manifest(inputs)
    by_name = {i["name"]: i for i in man.get("inputs", [])}
    rows = []
    for ctx in CONTEXTS:
        tag = f"{ctx // 1024}k"
        for domain in DOMAINS:
            name = f"{domain}-{tag}"
            path = Path(inputs) / f"{name}.bin"
            if not path.is_file():
                raise SystemExit(f"missing input {path}; run bench/build_inputs.py first")
            if name in by_name and by_name[name].get("sha256"):
                digest = by_name[name]["sha256"]
            else:
                digest = sha256_file(path)
            if domain == "prose":
                src = man.get("prose", {}).get("title")
                source = f"PG-19 {src}" if src else "PG-19 (public-domain book)"
            else:
                source = "llama.cpp src/*.cpp (MIT)"
            rows.append({"name": name, "domain": domain, "context": ctx, "path": path,
                         "tokens": ctx, "source": source, "sha256": digest})
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local EXL3 model directory")
    ap.add_argument("--inputs", default="bench/inputs", help="directory from build_inputs.py")
    ap.add_argument("--exllamav3", default=os.environ.get("EXLLAMAV3_PATH", "."),
                    help="exllamav3 checkout (added to sys.path)")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--revision", default=None, help="HF revision of the model snapshot")
    ap.add_argument("--model-repo", default="boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw")
    ap.add_argument("--bpw", type=float, default=2.10)
    ap.add_argument("--size-gb", type=float, default=153.6)
    ap.add_argument("--cpu-moe-layers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1, help="excluded warm-up passes (0 or 1)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.exllamav3).resolve()))
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    rows = input_rows(args.inputs)
    ctx_max = max(r["context"] for r in rows)

    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_offload = args.cpu_moe_layers
    config.infer_params.moe_cpu_threads = args.threads
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=ctx_max + args.gen_tokens + 256)
    model.load(progressbar=False)
    generator = Generator(model, cache, tokenizer, max_chunk_size=args.chunk)
    sampler = ArgmaxSampler()

    def run_one(ids, max_new):
        job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler)
        generator.enqueue(job)
        last = None
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("eos"):
                    last = r
        if last is None:
            raise RuntimeError("generation produced no EOS result")
        prefill = float(last["time_prefill"])
        gen_time = float(last["time_generate"])
        new = int(last["new_tokens"])
        return {
            "prefill_tps": ids.shape[1] / prefill if prefill > 0 else None,
            "decode_tps": (new - 1) / gen_time if gen_time > 0 and new > 1 else None,
            "ttft_s": prefill,
            "new_tokens": new,
        }

    if args.warmup:
        first = read_ids(rows[0]["path"])
        run_one(first, 16)
        torch.cuda.synchronize()

    cells = []
    for r in rows:
        ids = read_ids(r["path"])
        reps = []
        for _ in range(args.reps):
            reps.append(run_one(ids, args.gen_tokens))
        cells.append({
            "context": r["context"],
            "domain": r["domain"],
            "prefill_tps": mean([x["prefill_tps"] for x in reps]),
            "decode_tps": mean([x["decode_tps"] for x in reps]),
            "ttft_s": mean([x["ttft_s"] for x in reps]),
            "gen_tokens": args.gen_tokens,
            "reps": [{"prefill_tps": x["prefill_tps"], "decode_tps": x["decode_tps"],
                      "ttft_s": x["ttft_s"]} for x in reps],
        })
        print(f"{r['context']:>6} {r['domain']:<5} "
              f"{cells[-1]['prefill_tps']:.1f} prefill tok/s  "
              f"{cells[-1]['decode_tps']:.2f} decode tok/s  "
              f"{cells[-1]['ttft_s']:.2f} s TTFT", flush=True)

    commit, diff = git_state(args.exllamav3)
    hip = getattr(torch.version, "hip", None)
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
    result = {
        "schema": "ling31-r9700-bench/1",
        "date": date.today().isoformat(),
        "model": {"repo": args.model_repo, "revision": args.revision,
                  "bpw": args.bpw, "size_gb": args.size_gb},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine(),
            "ram": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3)} GB",
            "os": f"{platform.system()} {platform.release()}",
            "kernel": platform.release(),
            "rocm": hip,
            "torch": torch.__version__,
            "exllamav3": f"{commit} (+local diff sha256 {diff})" if diff else f"{commit or 'unknown'} (no local diff)",
        },
        "config": {
            "cpu_threads": args.threads,
            "chunk": args.chunk,
            "cpu_moe_layers": args.cpu_moe_layers,
            "env": env_snapshot(),
            "cmd": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        },
        "inputs": [{"name": r["name"], "source": r["source"], "tokens": r["tokens"],
                    "sha256": r["sha256"]} for r in rows],
        "cells": cells,
        "baseline": {"config": {}, "cells": []},
        "screen": [],
        "peak_vram_gb": vram,
        "peak_ram_gb": peak_rss_gb(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
