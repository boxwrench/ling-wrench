#!/usr/bin/env python3
"""Render the marker blocks in the release pages from results JSON.

Reads `results/results.json` (schema `ling31-r9700-bench/1`) and, when present,
`results/quality.json` (schema `ling31-wrench-quality/1`), then rewrites the
BENCH, RECEIPT and QUALITY blocks in:

  README.md
  results/README.md
  the Hugging Face card draft

The renderer is deterministic, so running it twice produces identical files.
`--check` exits non-zero if any managed block still contains `TBD`.
"""

import argparse
import json
import sys
from pathlib import Path

BENCH_SCHEMA = "ling31-r9700-bench/1"
QUALITY_SCHEMA = "ling31-wrench-quality/1"
BLOCKS = ("BENCH", "RECEIPT", "QUALITY")


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
    model = res.get("model") or {}
    layers = cfg.get("cpu_moe_layers")
    threads = cfg.get("cpu_threads")
    chunk = cfg.get("chunk")
    cfg_text = " / ".join(fmt_int(x) if x is not None else "TBD" for x in (layers, threads, chunk))
    env = cfg.get("env") or {}
    env_text = ", ".join(f"`{k}={v}`" for k, v in sorted(env.items())) if env else "none recorded"
    vram = fmt(res.get("peak_vram_gb"), 1, " GB")
    ram = fmt(res.get("peak_ram_gb"), 1, " GB")
    rows = [
        ("GPU", hw.get("gpu") or "TBD"),
        ("CPU", hw.get("cpu") or "TBD"),
        ("RAM", hw.get("ram") or "TBD"),
        ("OS", hw.get("os") or "TBD"),
        ("Kernel", hw.get("kernel") or "TBD"),
        ("ROCm", hw.get("rocm") or "TBD"),
        ("PyTorch", hw.get("torch") or "TBD"),
        ("exllamav3", hw.get("exllamav3") or "TBD"),
        ("Model revision", model.get("revision") or "TBD"),
        ("CPU-MoE layers / threads / chunk", cfg_text),
        ("Env", env_text),
        ("Command", f"`{cfg['cmd']}`" if cfg.get("cmd") else "TBD"),
        ("Peak VRAM / RAM", f"{vram} / {ram}"),
    ]
    lines = ["| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def render_quality(q):
    if not q:
        return None
    lines = [
        "| Candidate | Size | KL top-64 | Exact KL | KL_pred | Top-1 | Exact dNLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in q.get("candidates") or []:
        lines.append(
            f"| {c.get('name') or 'TBD'} | {fmt(c.get('size_gb'), 1, ' GB')} | "
            f"{fmt(c.get('kl_top64'), 3)} | {fmt(c.get('exact_kl'), 3)} | "
            f"{fmt(c.get('kl_pred'), 3)} | {fmt(c.get('top1_pct'), 2, ' %')} | "
            f"{fmt(c.get('dnll_exact'), 3)} |"
        )
    status = (q.get("status") or "").lower()
    verdict = q.get("verdict")
    if verdict and status in ("final", "complete", "done"):
        note = f"**Verdict.** {verdict}"
    else:
        note = ("**Verdict pending a noise-floor check.** The numbers above are single-run "
                "rankings; the same-configuration repeat noise floor is still being measured. "
                "Treat the ordering as provisional until it is reported.")
    floor = q.get("noise_floor")
    if floor is not None:
        note += f" Same-config repeat floor: {fmt(floor, 4)}."
    lines += ["", note]
    return "\n".join(lines)


def managed_files(root, card):
    files = [root / "README.md", root / "results" / "README.md"]
    if card and card.is_file():
        files.append(card)
    return files


def load_json(path, schema):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"cannot read {path}: {e}")
    if data.get("schema") != schema:
        raise SystemExit(f"{path}: expected schema {schema!r}, got {data.get('schema')!r}")
    return data


def blocks_in(text):
    out = {}
    for name in BLOCKS:
        start = f"<!-- {name}:START -->"
        end = f"<!-- {name}:END -->"
        if start in text and end in text:
            out[name] = text.split(start, 1)[1].split(end, 1)[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=root_default, help="repository root")
    ap.add_argument("--results", type=Path, default=None,
                    help="results JSON (default: <root>/results/results.json)")
    ap.add_argument("--quality", type=Path, default=None,
                    help="quality JSON (default: <root>/results/quality.json if present)")
    ap.add_argument("--card", type=Path, default=None,
                    help="optional HF card draft to render; skipped when absent")
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if a managed block still contains TBD")
    args = ap.parse_args()

    root = args.root.resolve()
    results_path = args.results or (root / "results" / "results.json")
    quality_path = args.quality or (root / "results" / "quality.json")
    card = args.card

    res = load_json(results_path, BENCH_SCHEMA)
    quality = None
    if quality_path.is_file():
        quality = load_json(quality_path, QUALITY_SCHEMA)

    files = managed_files(root, card)
    if not files:
        raise SystemExit("no managed files found")

    bodies = {
        "BENCH": render_bench(res),
        "RECEIPT": render_receipt(res),
    }
    qbody = render_quality(quality)
    if qbody is not None:
        bodies["QUALITY"] = qbody

    if args.check:
        bad = []
        seen = 0
        for f in files:
            if not f.is_file():
                bad.append(f"{f}: missing")
                continue
            text = f.read_text(encoding="utf-8")
            found = blocks_in(text)
            for name, body in found.items():
                seen += 1
                if "TBD" in body:
                    bad.append(f"{f}: {name} block still has TBD")
            if not found:
                bad.append(f"{f}: no marker blocks")
        if bad:
            print("check failed:", file=sys.stderr)
            for b in bad:
                print("  " + b, file=sys.stderr)
            return 1
        print(f"check passed: {seen} blocks, no TBD")
        return 0

    changed = 0
    for f in files:
        if not f.is_file():
            print(f"skip missing {f}", file=sys.stderr)
            continue
        text = f.read_text(encoding="utf-8")
        new = text
        used = []
        for name, body in bodies.items():
            new, hit = replace_block(new, name, body)
            if hit:
                used.append(name)
        if new != text:
            f.write_text(new, encoding="utf-8")
            changed += 1
        print(f"{f}: {', '.join(used) if used else 'no blocks'}")
    print(f"updated {changed} file(s)")

    # A fill run should leave everything renderable; warn, but do not fail, if
    # the source JSON itself has no numbers yet.
    if "TBD" in render_bench(res) or "TBD" in render_receipt(res):
        print("note: speed/receipt placeholders remain (results.json has null fields)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
