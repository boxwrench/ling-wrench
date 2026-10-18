#!/usr/bin/env python3
import re, sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
root = Path(__file__).resolve().parents[1]
files = sorted(root.rglob('*.md')) + [root / 'llms.txt']
errors, checked = [], 0
for source in files:
    text = source.read_text()
    targets = re.findall(r'\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+"[^"]*")?\)', text) + re.findall(r'^\[[^\]]+\]:\s*<?([^\s>]+)>?', text, re.M) + re.findall(r'(?:href|src)=["\']([^"\']+)', text)
    targets += ['AGENTS.md'] if source.name == 'CLAUDE.md' else []
    for target in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc: continue
        dest = (source.parent / unquote(url.path)).resolve() if url.path else source
        if dest.is_dir(): dest = dest / 'README.md' if url.fragment else dest
        checked += 1
        anchors = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', dest.read_text(), re.M)} if dest.is_file() and url.fragment else set()
        if not dest.exists() or (url.fragment and unquote(url.fragment) not in anchors): errors.append(f'{source.relative_to(root)}: {target}')
print('\n'.join(errors) if errors else f'links: {checked} checked, 0 broken'); sys.exit(bool(errors))
