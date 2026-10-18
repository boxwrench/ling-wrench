#!/usr/bin/env python3
"""Build the public benchmark inputs for Ling-3.1.

Prose: the first book in the PG-19 test split that tokenizes to at least
70,000 Ling tokens. Code: sorted llama.cpp `src/*.cpp` concatenated in order.

Both are tokenized exactly as the exllamav3 runtime does it: the model BOS
token is prepended, and strings that look like special tokens (for example
`[gMASK]`) are encoded as plain text (`encode_special_tokens`). The result is
cut to exactly 8,192 and 65,536 ids; the 8K window is the prefix of the 64K
window. Token ids are written as little-endian uint32 `.bin` files, with a
SHA-256 for each file (identical bytes to the int32 files for these ids).

Pinned sources: PG-19 test (`emozilla/pg19` revision
c021754c8e01c5b1cc83a1f549c1f97fbbb756b8, file
`data/test-00000-of-00001-29a571947c0b5ccc.parquet`, sha256
9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc) and
llama.cpp `e467e5b5e5ba43071d6151b3d881deb9300d000e` (MIT). Requires
`tokenizers` and, for a local parquet file, `pyarrow`; the `hf://` form also
needs `huggingface_hub`.

No private text is read by this script.
"""

import argparse
import glob
import hashlib
import json
import struct
import sys
from pathlib import Path

TARGETS = (8192, 65536)
MIN_BOOK_TOKENS = 70000
PG19_REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tokenizer(path):
    from tokenizers import Tokenizer

    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer.json"
    if not p.is_file():
        raise SystemExit(f"tokenizer.json not found at {p}")
    tok = Tokenizer.from_file(str(p))
    # Match exllamav3: special-token-looking substrings are plain text.
    tok.encode_special_tokens = True

    bos_id = None
    cfg = p.parent / "tokenizer_config.json"
    if cfg.is_file():
        bos = json.loads(cfg.read_text(encoding="utf-8")).get("bos_token")
        if isinstance(bos, dict):
            bos = bos.get("content")
        if isinstance(bos, str) and bos:
            bos_id = tok.token_to_id(bos)
            if bos_id is None:
                raise SystemExit(f"BOS token {bos!r} is not in {p}")
    return tok, bos_id


def encode(tok, text, bos_id):
    ids = tok.encode(text, add_special_tokens=False).ids
    return ([bos_id] + ids) if bos_id is not None else ids


def pg19_batches(source, split, revision=PG19_REVISION):
    """Yield (title, url, text) from a local parquet file/dir or hf://repo@split."""
    import pyarrow.parquet as pq

    files = []
    if source.startswith("hf://"):
        from huggingface_hub import HfApi, hf_hub_download

        spec = source[len("hf://"):]
        repo, _, hf_split = spec.partition("@")
        hf_split = hf_split or split
        names = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
        names = sorted(n for n in names if n.startswith(f"data/{hf_split}-") and n.endswith(".parquet"))
        if not names:
            raise SystemExit(f"no parquet shards for split {hf_split!r} in {repo}")
        for n in names:
            files.append(hf_hub_download(repo, n, repo_type="dataset", revision=revision))
    else:
        p = Path(source)
        if p.is_dir():
            files = sorted(glob.glob(str(p / "**" / "*.parquet"), recursive=True))
        elif p.is_file():
            files = [str(p)]
        if not files:
            raise SystemExit(f"no parquet files under {source}")

    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in ("short_book_title", "url", "text") if c in pf.schema_arrow.names]
        if "text" not in cols:
            raise SystemExit(f"{f} has no 'text' column")
        for batch in pf.iter_batches(columns=cols, batch_size=8):
            for row in batch.to_pylist():
                yield row.get("short_book_title") or "(untitled)", row.get("url") or "", row["text"]


def pick_prose(source, split, tok, bos_id, need, min_tokens=MIN_BOOK_TOKENS, revision=PG19_REVISION):
    for title, url, text in pg19_batches(source, split, revision):
        ids = encode(tok, text, bos_id)
        if len(ids) >= min_tokens:
            print(f"prose: {title!r} ({len(ids):,} tokens) -> {need:,}", file=sys.stderr)
            return title, url, ids[:need]
    raise SystemExit(f"no PG-19 book in {source} reached {need:,} tokens")


def pick_code(code_dir, tok, bos_id, need):
    root = Path(code_dir)
    src = root / "src"
    files = sorted(glob.glob(str((src if src.is_dir() else root) / "*.cpp")))
    if not files:
        raise SystemExit(f"no *.cpp files under {src if src.is_dir() else root}")
    # Encode the whole concatenation once, as the benchmark did; per-file
    # encoding can split bytes differently at the joins.
    parts = []
    used = []
    for f in files:
        text = f"\n\n// ===== FILE: {Path(f).name} =====\n\n"
        body = Path(f).read_text(encoding="utf-8", errors="replace")
        parts.append(text)
        parts.append(body)
        used.append({"file": Path(f).name})
    ids = encode(tok, "".join(parts), bos_id)
    if len(ids) < need:
        raise SystemExit(f"code source reached only {len(ids):,} tokens, need {need:,}")
    print(f"code: {len(used)} files, {len(ids):,} tokens -> {need:,}", file=sys.stderr)
    return used, ids[:need]


def write_ids(out_dir, name, ids):
    bin_path = out_dir / f"{name}.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}I", *ids))
    txt_path = out_dir / f"{name}.txt"
    txt_path.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
    return bin_path, sha256_file(bin_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True,
                    help="Ling-3.1 model directory or tokenizer.json")
    ap.add_argument("--pg19", default="hf://emozilla/pg19@test",
                    help="local PG-19 parquet/dir, or hf://repo@split (default: emozilla/pg19 "
                         "test; pin revision c021754c8e01c5b1cc83a1f549c1f97fbbb756b8)")
    ap.add_argument("--split", default="test", help="split used when --pg19 has no @split")
    ap.add_argument("--pg19-revision", default=PG19_REVISION,
                    help="dataset revision for hf:// downloads (default: pinned E239 revision)")
    ap.add_argument("--code-dir", default="llama.cpp",
                    help="llama.cpp checkout at e467e5b5e5ba43071d6151b3d881deb9300d000e; "
                         "src/*.cpp is used if present")
    ap.add_argument("--out", default="bench/inputs", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, bos_id = load_tokenizer(args.tokenizer)

    need = max(TARGETS)
    title, url, prose_ids = pick_prose(args.pg19, args.split, tok, bos_id, need, MIN_BOOK_TOKENS,
                                     args.pg19_revision)
    code_files, code_ids = pick_code(args.code_dir, tok, bos_id, need)

    manifest = {
        "tokenizer": "Ling-3.1 tokenizer.json (from the model snapshot)",
        "targets": list(TARGETS),
        "prose": {"source": "PG-19", "split": args.split, "title": title, "url": url},
        "code": {"source": "llama.cpp", "license": "MIT", "files": code_files},
        "inputs": [],
    }

    sums = []
    for domain, ids in (("prose", prose_ids), ("code", code_ids)):
        for n in TARGETS:
            name = f"{domain}-{n // 1024}k"
            path, digest = write_ids(out, name, ids[:n])
            manifest["inputs"].append({
                "name": name, "domain": domain, "tokens": n,
                "file": path.name, "sha256": digest,
            })
            sums.append(f"{digest}  {path.name}")
            print(f"{name}  {n}  {digest}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    print(f"wrote {out}/SHA256SUMS and {out}/manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
