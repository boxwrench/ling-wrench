#!/usr/bin/env python3
"""exllamav3 runner for the Ling-3.1 one-R9700 benchmark.

Runs the 8K and 64K cells (prose and code), greedy, 256 new tokens, two
measured repetitions after an excluded 16-token warm-up *per cell*. Each
context is measured in its own process so the CPU-MoE layer count can differ
(60 at 8K, 62 at 64K on the reference R9700); a single ``--contexts`` value
runs in this process.

Output conforms to schema `ling-r9700-bench/1`; the release pages are rendered
from it by `tools/fill_results.py`. No private paths are used.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

DOMAINS = ("prose", "code")
KNOWN_CONTEXTS = (8192, 65536)
DEFAULT_GEN = 256
WARMUP_TOKENS = 16


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
    commit = sh(f"git -C {path} rev-parse HEAD")
    diff = sh(f"git -C {path} diff")
    digest = hashlib.sha256(diff.encode()).hexdigest() if diff else None
    return commit, digest


def os_pretty():
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return f"{platform.system()} {platform.release()}"


def mem_total_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return None


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return None


def env_snapshot():
    keys = ("CUDA_VISIBLE_DEVICES", "ROCM_HOME", "PYTORCH_ROCM_ARCH", "HIP_LAUNCH_BLOCKING",
            "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            "EXL3_MOE_STREAM_T", "EXL3_MOE_STREAM_FUSED_T", "EXL3_MOE_RECON_DET",
            "EXL3_MOE_CPU_THREADS", "EXL3_MOE_CPU_OFFLOAD", "EXL3_MOE_HOST_CORES",
            "EXL3_MOE_CPU_SWAP", "EXL3_FE_GEMM")
    return {k: os.environ[k] for k in keys if k in os.environ}


def load_manifest(inputs):
    p = Path(inputs) / "manifest.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def input_rows(inputs, contexts):
    man = load_manifest(inputs)
    by_name = {i["name"]: i for i in man.get("inputs", [])}
    rows = []
    for ctx in contexts:
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


def parse_contexts(text):
    try:
        contexts = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--contexts must be comma-separated integers, got {text!r}")
    if not contexts:
        raise SystemExit("--contexts is empty")
    bad = [c for c in contexts if c not in KNOWN_CONTEXTS]
    if bad:
        raise SystemExit(f"unsupported context(s) {bad}; known: {list(KNOWN_CONTEXTS)}")
    return contexts


def parse_layers(text, contexts):
    try:
        values = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--cpu-moe-layers must be an integer or comma-separated integers, got {text!r}")
    if len(values) == 1:
        values = values * len(contexts)
    if len(values) != len(contexts):
        raise SystemExit(f"--cpu-moe-layers has {len(values)} value(s) for {len(contexts)} context(s); "
                         "give one value or one per context")
    if any(v < 0 for v in values):
        raise SystemExit("--cpu-moe-layers must be >= 0")
    return dict(zip(contexts, values))


def run_shard(args, context, layers):
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    rows = input_rows(args.inputs, [context])
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_offload = layers
    config.infer_params.moe_cpu_threads = args.threads
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=context + 1024, max_batch_size=1, max_history=0)
    load_start = time.perf_counter()
    print(json.dumps({"event": "load_start", "context": context, "cpu_moe_layers": layers}), flush=True)
    model.load(device=args.device, progressbar=False, max_chunk_size=args.chunk, max_output_size=1)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    print(json.dumps({"event": "loaded", "context": context, "load_seconds": load_seconds}), flush=True)
    sampler = ArgmaxSampler()

    stop = threading.Event()
    mem = {"min_available_gb": mem_available_gb()}

    def monitor():
        while not stop.is_set():
            v = mem_available_gb()
            if v is not None:
                mem["min_available_gb"] = min(mem["min_available_gb"], v)
            stop.wait(0.25)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    def run_one(ids, max_new):
        # Fresh generator per pass: no paged-prefix or recurrent-state reuse.
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1,
                        max_chunk_size=args.chunk, ngram_match_min=0, recurrent_cache_size=0)
        job = Job(input_ids=ids, max_new_tokens=max_new, stop_conditions=[],
                  sampler=sampler, decode_special_tokens=True)
        gen.enqueue(job)
        torch.cuda.synchronize()
        final = None
        try:
            with torch.inference_mode():
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        if r.get("eos"):
                            final = r
            new_ids = job.sequences[0].sequence_ids.torch()[0, ids.shape[1]:].tolist()
            if len(new_ids) != max_new:
                raise RuntimeError(f"expected {max_new} generated ids, got {len(new_ids)}")
        finally:
            gen.close()
            del gen, job
            gc.collect()
            torch.cuda.empty_cache()
        if final is None:
            raise RuntimeError("generation produced no final result")
        prefill = float(final["time_prefill"])
        gen_time = float(final["time_generate"])
        new = int(final["new_tokens"])
        if new != max_new:
            raise RuntimeError(f"expected {max_new} generated tokens, got {new}")
        if int(final["cached_tokens"]) != 0:
            raise RuntimeError(f"unexpected prefix reuse: {final['cached_tokens']} cached tokens")
        return {
            "prefill_tps": ids.shape[1] / prefill if prefill > 0 else None,
            "decode_tps": (new - 1) / gen_time if gen_time > 0 and new > 1 else None,
            "ttft_s": prefill,
            "new_tokens": new,
            "cached_tokens": int(final["cached_tokens"]),
            "cached_pages": int(final["cached_pages"]),
            "prompt_tokens": int(final["prompt_tokens"]),
            "ids_sha256_u32": hashlib.sha256(struct.pack(f"<{len(new_ids)}I", *new_ids)).hexdigest(),
            "ids": new_ids,
        }

    cells = []
    try:
        for r in rows:
            ids = read_ids(r["path"])
            if args.warmup:
                run_one(ids, WARMUP_TOKENS)
            reps = []
            for rep in range(args.reps):
                result = run_one(ids, args.gen_tokens)
                reps.append(result)
                print(json.dumps({"event": "rep", "context": context, "domain": r["domain"],
                                  "rep": rep, **{k: v for k, v in result.items() if k != "ids"}}), flush=True)
            cells.append({
                "context": r["context"],
                "domain": r["domain"],
                "cpu_moe_layers": layers,
                "prefill_tps": mean([x["prefill_tps"] for x in reps]),
                "decode_tps": mean([x["decode_tps"] for x in reps]),
                "ttft_s": mean([x["ttft_s"] for x in reps]),
                "gen_tokens": args.gen_tokens,
                "reps": reps,
            })
            print(f"{r['context']:>6} {r['domain']:<5} "
                  f"{cells[-1]['prefill_tps']:.1f} prefill tok/s  "
                  f"{cells[-1]['decode_tps']:.2f} decode tok/s  "
                  f"{cells[-1]['ttft_s']:.2f} s TTFT", flush=True)
    finally:
        stop.set()
        th.join(timeout=1)

    commit, diff = git_state(args.exllamav3)
    total = mem_total_gb()
    avail = mem["min_available_gb"]
    peak_ram = (total - avail) if total is not None and avail is not None else None
    hip = getattr(torch.version, "hip", None)
    vram = torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else None
    cfg = {
        "cpu_threads": args.threads,
        "load_device": args.device or "autosplit",
        "chunk": args.chunk,
        "cpu_moe_layers": layers,
        "contexts": [context],
        "env": env_snapshot(),
        f"cpu_moe_layers_{context // 1024}k": layers,
        "cmd": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
    }
    return {
        "schema": "ling-r9700-bench/1",
        "kind": "exl3-bench",
        "date": date.today().isoformat(),
        "model": {"repo": args.model_repo, "revision": args.revision,
                  "bpw": args.bpw, "size_gb": args.size_gb},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine(),
            "ram": f"{round(total)} GB" if total else "unknown",
            "os": os_pretty(),
            "kernel": platform.release(),
            "rocm": hip,
            "torch": torch.__version__,
            "exllamav3": f"{commit} (+local diff sha256 {diff})" if diff else f"{commit or 'unknown'} (no local diff)",
        },
        "config": cfg,
        "inputs": [{"name": r["name"], "source": r["source"], "tokens": r["tokens"],
                    "sha256": r["sha256"]} for r in rows],
        "cells": cells,
        "baseline": {"config": {}, "cells": []},
        "screen": [],
        "peak_vram_gb": vram,
        "peak_ram_gb": peak_ram,
        "load_seconds": load_seconds,
    }


def child_cmd(args, context, layers, out):
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--model", args.model,
           "--inputs", args.inputs,
           "--exllamav3", args.exllamav3,
           "--out", str(out),
           "--contexts", str(context),
           "--cpu-moe-layers", str(layers),
           "--threads", str(args.threads),
           "--chunk", str(args.chunk),
           "--gen-tokens", str(args.gen_tokens),
           "--reps", str(args.reps),
           "--warmup", str(args.warmup),
           "--model-repo", args.model_repo,
           "--bpw", str(args.bpw),
           "--size-gb", str(args.size_gb)]
    if args.revision:
        cmd += ["--revision", args.revision]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def public_cmd(args, contexts, layers):
    cmd = ["bench/run_bench.py", "--model", "$MODEL_DIR", "--inputs", "$INPUTS",
           "--exllamav3", "$EXLLAMAV3",
           "--contexts", ",".join(map(str, contexts)),
           "--cpu-moe-layers", ",".join(str(layers[c]) for c in contexts),
           "--threads", str(args.threads), "--chunk", str(args.chunk),
           "--reps", str(args.reps), "--warmup", str(args.warmup),
           "--out", "ling-3.1/results/results.json"]
    if args.device:
        cmd += ["--device", args.device]
    return " ".join(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local EXL3 model directory")
    ap.add_argument("--inputs", default="bench/inputs", help="directory from build_inputs.py")
    ap.add_argument("--exllamav3", default=os.environ.get("EXLLAMAV3_PATH", "."),
                    help="exllamav3 checkout (added to sys.path)")
    ap.add_argument("--out", default="ling-3.1/results/results.json")
    ap.add_argument("--revision", default=None, help="HF revision of the model snapshot")
    ap.add_argument("--model-repo", default="boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw")
    ap.add_argument("--bpw", type=float, default=2.10)
    ap.add_argument("--size-gb", type=float, default=153.6)
    ap.add_argument("--contexts", default="8192,65536",
                    help="comma-separated contexts; known: 8192,65536 (default: both)")
    ap.add_argument("--cpu-moe-layers", required=True,
                    help="routed-expert layers offloaded to the CPU worker (-mcl): one value for "
                         "all contexts, or one per context in --contexts order (e.g. 60,62)")
    ap.add_argument("--device", default=None,
                    help="explicit single device (e.g. cuda:0); omitted uses autosplit")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4096, help="maximum prompt chunk (-gcs)")
    ap.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1,
                    help="excluded 16-token warm-up per cell (0 or 1)")
    args = ap.parse_args()

    if args.warmup not in (0, 1):
        raise SystemExit("--warmup must be 0 or 1")
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    contexts = parse_contexts(args.contexts)
    layers = parse_layers(args.cpu_moe_layers, contexts)

    sys.path.insert(0, str(Path(args.exllamav3).resolve()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(contexts) == 1:
        result = run_shard(args, contexts[0], layers[contexts[0]])
    else:
        # One process per context, as the reference run did: the CPU-MoE layer
        # assignment is fixed at model load, so 60 (8K) and 62 (64K) cannot
        # share one loaded model.
        parts = []
        try:
            for ctx in contexts:
                part = out.parent / f".{out.stem}.{ctx}.partial.json"
                subprocess.run(child_cmd(args, ctx, layers[ctx], part), check=True)
                parts.append(json.loads(part.read_text()))
                part.unlink()
        finally:
            for p in out.parent.glob(f".{out.stem}.*.partial.json"):
                p.unlink(missing_ok=True)
        result = dict(parts[0])
        result["cells"] = [c for p in parts for c in p["cells"]]
        result["inputs"] = [i for p in parts for i in p["inputs"]]
        result["peak_vram_gb"] = max(p.get("peak_vram_gb") or 0 for p in parts)
        result["peak_ram_gb"] = max(p.get("peak_ram_gb") or 0 for p in parts)
        cfg = dict(parts[0]["config"])
        cfg.pop("contexts", None)
        cfg.pop("cpu_moe_layers_8k", None)
        cfg.pop("cpu_moe_layers_64k", None)
        cfg["contexts"] = contexts
        cfg["cpu_moe_layers"] = layers[contexts[0]] if len(set(layers.values())) == 1 else None
        for ctx, n in layers.items():
            cfg[f"cpu_moe_layers_{ctx // 1024}k"] = n
        cfg["cmd"] = public_cmd(args, contexts, layers)
        result["config"] = cfg
        result["date"] = date.today().isoformat()

    result["config"]["cmd"] = public_cmd(args, contexts, layers)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
