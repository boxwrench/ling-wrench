#!/usr/bin/env python3
"""Render the release pages from their results JSON.

Both releases use the schema `ling-r9700-bench/1`, with a `kind` field:

  * `exl3-bench` (Ling-3.1) - BENCH, RECEIPT and QUALITY blocks.
  * `serving-check` (Ling-3.0) - RUNTIME and RECEIPT blocks; a card
    selects one checkpoint with an HTML MODEL_REPO marker.

Reads `ling-3.1/results/results.json` and/or `ling-3.0/results/runtime.json`,
plus the optional 3.1 `results/quality.json`, then rewrites the marker blocks in
the matching pages. The renderer is deterministic: running it twice produces
identical files.

`--check` exits non-zero if a managed block still contains `TBD`.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "ling-r9700-bench/1"
QUALITY_SCHEMA = "ling31-wrench-quality/1"
SCHEMA_ALIASES = {"ling31-r9700-bench/1": SCHEMA}  # name the 3.1 benchmark brief wrote
BLOCKS = {
    "3.1": ("BENCH", "RECEIPT", "QUALITY"),
    "3.0": ("RUNTIME", "RECEIPT"),
}


def fmt(v, nd=1, suffix=""):
    if v is None:
        return "TBD"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return f"{v:,}{suffix}"
    return f"{v:.{nd}f}{suffix}"


def fmt_int(v):
    if v is None:
        return "TBD"
    return f"{int(v):,}"


def profile_of(res):
    kind = (res.get("kind") or "").lower()
    if kind in ("serving-check", "serving", "runtime") or res.get("models"):
        return "3.0"
    return "3.1"


def replace_block(text, name, body):
    start = f"<!-- {name}:START -->"
    end = f"<!-- {name}:END -->"
    if start not in text or end not in text:
        return text, False
    pre, rest = text.split(start, 1)
    _old, post = rest.split(end, 1)
    new = pre + start + "\n" + body.rstrip() + "\n" + end + post
    return new, True


def render_bench(res):
    cells = {}
    for c in res.get("cells") or []:
        cells[(c.get("context"), c.get("domain"))] = c
    lines = [
        "| Context | Domain | Prefill tok/s | Decode tok/s | TTFT (s) |",
        "|---:|---|---:|---:|---:|",
    ]
    for ctx in (8192, 65536):
        for domain in ("prose", "code"):
            c = cells.get((ctx, domain), {})
            lines.append(
                f"| {ctx // 1024}K | {domain} | "
                f"{fmt(c.get('prefill_tps'), 1)} | "
                f"{fmt(c.get('decode_tps'), 2)} | "
                f"{fmt(c.get('ttft_s'), 2)} |"
            )
    return "\n".join(lines)


def render_receipt(res):
    hw = res.get("hardware") or {}
    cfg = res.get("config") or {}
    layers = cfg.get("cpu_moe_layers")
    if cfg.get("cpu_moe_layers_8k") is not None and cfg.get("cpu_moe_layers_64k") is not None:
        layers = f"{cfg['cpu_moe_layers_8k']} (8K) or {cfg['cpu_moe_layers_64k']} (64K)"
    threads = cfg.get("cpu_threads")
    chunk = cfg.get("chunk")
    cfg_text = " / ".join(x if isinstance(x, str) else fmt_int(x) if x is not None else "TBD"
                          for x in (layers, threads, chunk))
    env = cfg.get("env") or {}
    env_text = ", ".join(f"`{k}={v}`" for k, v in sorted(env.items())) if env else "none"
    vram = fmt(res.get("peak_vram_gb"), 1, " GiB")
    ram = fmt(res.get("peak_ram_gb"), 1, " GiB")
    rows = [
        ("Hardware", f"{hw.get('gpu') or 'TBD'}; {hw.get('cpu') or 'TBD'}; {hw.get('ram') or 'TBD'}"),
        ("Software", f"{hw.get('os') or 'TBD'}; kernel {hw.get('kernel') or 'TBD'}; "
                     f"ROCm {hw.get('rocm') or 'TBD'}; PyTorch {hw.get('torch') or 'TBD'}; "
                     f"exllamav3 {hw.get('exllamav3') or 'TBD'}"),
        ("Run", f"CPU-MoE layers / threads / chunk {cfg_text}; env {env_text}; "
                f"peak VRAM / RAM {vram} / {ram}"),
    ]
    lines = ["| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def render_serving_receipt(res):
    hw = res.get("hardware") or {}
    rt = res.get("runtime") or {}
    rows = [
        ("Hardware", f"{hw.get('gpu') or 'TBD'}; {hw.get('cpu') or 'TBD'}; {hw.get('ram') or 'TBD'}"),
        ("Software", f"{rt.get('name') or 'TBD'} {rt.get('version') or ''}".strip()
                     + f"; {rt.get('backend') or 'TBD'}; {rt.get('model_format') or 'TBD'}"),
        ("Method", res.get("protocol") or "TBD"),
    ]
    lines = ["| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def _vals(runs, key):
    out = []
    for r in runs:
        v = r.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _span(vals, nd):
    if not vals:
        return "TBD"
    med = statistics.median(vals)
    if len(vals) == 1:
        return f"{med:.{nd}f}"
    return f"{med:.{nd}f} [{min(vals):.{nd}f}-{max(vals):.{nd}f}]"


def render_runtime(res, include_protocol=True):
    lines = [
        "| Model | Status | Prompt | Prefill tok/s | Decode tok/s | TTFT (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in res.get("models") or []:
        name = m.get("name") or "TBD"
        status = m.get("status") or "TBD"
        for p in m.get("points") or []:
            target = p.get("target")
            label = f"{int(target) // 1024}K" if isinstance(target, int) else "TBD"
            runs = p.get("runs") or []
            pp = _span(_vals(runs, "prefill_tps"), 1)
            tg = _span(_vals(runs, "decode_tps"), 2)
            tt = _span(_vals(runs, "ttft_s"), 2)
            lines.append(f"| {name} | {status} | {label} | {pp} | {tg} | {tt} |")
    note = res.get("protocol")
    if note and include_protocol:
        lines += ["", f"Median of the runs; brackets are min-max. {note}"]
    return "\n".join(lines)


def render_quality(q):
    if not q:
        return None
    lines = [
        "| Candidate | Size | KL top-64 | Top-1 |",
        "|---|---:|---:|---:|",
    ]
    for c in q.get("candidates") or []:
        lines.append(
            f"| {c.get('name') or 'TBD'} | {fmt(c.get('size_gb'), 1, ' GB')} | "
            f"{fmt(c.get('kl_top64'), 3)} | {fmt(c.get('top1_pct'), 2, ' %')} |"
        )
    status = (q.get("status") or "").lower()
    verdict = q.get("verdict")
    if verdict and status in ("final", "complete", "done"):
        note = f"**Verdict.** {verdict}"
    else:
        note = ("**Provisional.** Single runs; KL is not task accuracy. The same-config "
                "repeat floor is still being measured.")
    floor = q.get("noise_floor")
    if floor is not None:
        note += f" Same-config repeat floor: {fmt(floor, 4)}."
    lines += ["", note]
    return "\n".join(lines)


def bodies_for(profile, res, quality):
    if profile == "3.1":
        bodies = {"BENCH": render_bench(res), "RECEIPT": render_receipt(res)}
        qbody = render_quality(quality)
        if qbody is not None:
            bodies["QUALITY"] = qbody
        return bodies
    return {"RUNTIME": render_runtime(res), "RECEIPT": render_serving_receipt(res)}


def managed_files(job):
    page = job["page"]
    files = [page]
    sibling = page.parent / "results" / "README.md"
    if sibling.is_file() and sibling != page:
        files.append(sibling)
    card = job.get("card")
    if card and card.is_file():
        files.append(card)
    return files


def blocks_in(text, names):
    out = {}
    for name in names:
        start = f"<!-- {name}:START -->"
        end = f"<!-- {name}:END -->"
        if start in text and end in text:
            out[name] = text.split(start, 1)[1].split(end, 1)[0]
    return out


def load_json(path, schema):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"cannot read {path}: {e}")
    if data.get("schema") != schema and SCHEMA_ALIASES.get(data.get("schema")) != schema:
        raise SystemExit(f"{path}: expected schema {schema!r}, got {data.get('schema')!r}")
    return data


def default_jobs(root):
    jobs = []
    p31 = root / "ling-3.1" / "results" / "results.json"
    if p31.is_file():
        jobs.append({
            "profile": "3.1",
            "results": p31,
            "quality": root / "ling-3.1" / "results" / "quality.json",
            "page": root / "ling-3.1" / "README.md",
        })
    p30 = root / "ling-3.0" / "results" / "runtime.json"
    if p30.is_file():
        jobs.append({
            "profile": "3.0",
            "results": p30,
            "quality": None,
            "page": root / "ling-3.0" / "README.md",
        })
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=root_default, help="repository root")
    ap.add_argument("--profile", choices=("3.1", "3.0"), default=None,
                    help="only this release (default: every results file found)")
    ap.add_argument("--results", type=Path, default=None,
                    help="one results JSON; disables the default two-release pass")
    ap.add_argument("--quality", type=Path, default=None,
                    help="3.1 quality JSON (default: next to the results file)")
    ap.add_argument("--page", type=Path, default=None,
                    help="page to render with --results (default: <root>/README.md)")
    ap.add_argument("--card", type=Path, default=None,
                    help="optional HF card draft; 3.0 cards need a MODEL_REPO marker")
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if a managed block still contains TBD")
    args = ap.parse_args()

    root = args.root.resolve()
    if args.results:
        res = load_json(args.results, SCHEMA)
        profile = args.profile or profile_of(res)
        if args.profile and args.profile != profile_of(res):
            raise SystemExit(f"{args.results}: profile {profile_of(res)} does not match --profile {args.profile}")
        quality = args.quality
        if quality is None and profile == "3.1":
            candidate = args.results.parent / "quality.json"
            quality = candidate if candidate.is_file() else None
        jobs = [{
            "profile": profile,
            "results": args.results,
            "quality": quality,
            "page": args.page or (root / "README.md"),
            "card": args.card,
        }]
    else:
        jobs = default_jobs(root)
        if args.profile:
            jobs = [j for j in jobs if j["profile"] == args.profile]
        if args.card:
            for j in jobs:
                if j["profile"] == (args.profile or "3.1"):
                    j["card"] = args.card
        if not jobs:
            raise SystemExit(f"no results JSON under {root}")

    # Load and render once per job.
    for job in jobs:
        job["res"] = job["results"] and load_json(job["results"], SCHEMA)
        if job["res"] is None:
            raise SystemExit(f"missing results for {job['profile']}")
        q = job.get("quality")
        job["quality_data"] = load_json(q, QUALITY_SCHEMA) if q and Path(q).is_file() else None
        job["bodies"] = bodies_for(job["profile"], job["res"], job["quality_data"])
        job["files"] = managed_files(job)

    if args.check:
        bad = []
        seen = 0
        for job in jobs:
            names = BLOCKS[job["profile"]]
            for f in job["files"]:
                if not f.is_file():
                    bad.append(f"{f}: missing")
                    continue
                text = f.read_text(encoding="utf-8")
                found = blocks_in(text, names)
                seen += len(found)
                if not found:
                    bad.append(f"{f}: no {job['profile']} marker blocks")
                if "TBD" in text:
                    hits = [name for name, body in found.items() if "TBD" in body]
                    bad.append(f"{f}: TBD in {', '.join(hits) if hits else 'text'}")
        if bad:
            print("check failed:", file=sys.stderr)
            for b in bad:
                print("  " + b, file=sys.stderr)
            return 1
        print(f"check passed: {seen} blocks, no TBD")
        return 0

    changed = 0
    for job in jobs:
        for f in job["files"]:
            if not f.is_file():
                print(f"skip missing {f}", file=sys.stderr)
                continue
            text = f.read_text(encoding="utf-8")
            new = text
            used = []
            bodies = job["bodies"]
            if job["profile"] == "3.0" and f == job.get("card"):
                marker = re.search(r"<!-- MODEL_REPO:([^\s]+) -->", text)
                models = [m for m in job["res"].get("models", [])
                          if marker and m.get("repo") == marker.group(1)]
                if len(models) != 1:
                    raise SystemExit("3.0 card needs MODEL_REPO matching exactly one results checkpoint")
                bodies = {"RUNTIME": render_runtime(dict(job["res"], models=models), include_protocol=False)}
            for name, body in bodies.items():
                new, hit = replace_block(new, name, body)
                if hit:
                    used.append(name)
            if new != text:
                f.write_text(new, encoding="utf-8")
                changed += 1
            print(f"{f}: {', '.join(used) if used else 'no blocks'}")
    print(f"updated {changed} file(s)")

    # Warn, but do not fail, when the source JSON itself has no numbers yet.
    for job in jobs:
        if any("TBD" in body for body in job["bodies"].values()):
            print(f"note: {job['profile']} has placeholders left (null fields in its results JSON)",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
