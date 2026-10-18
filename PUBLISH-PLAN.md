# Publish plan — owner-run only

Everything here is prepared locally. **Do not run these commands from an agent
session.** The owner reviews the diff and the rendered pages, then runs them.

## 0. Preconditions

- The benchmark has written `results/results.json`, or this is a deliberate
  pages-only release with the placeholders left in place.
- The quality verdict, if it has landed, is in `results/quality.json`
  (`status` and `verdict` set).
- The Hugging Face card draft is at `$CARD` (the draft lives outside this
  repository so the public tree does not link to a draft path).

```sh
cd ling31-wrench-public
export CARD=/path/to/ling31-wrench-hf-README.md
```

## 1. Render and validate

```sh
python tools/test_fill_results.py

# Fill the marker blocks from results/results.json and results/quality.json.
python tools/fill_results.py --root . --card "$CARD"

# Must exit 0. It fails if any BENCH, RECEIPT or QUALITY block still has TBD.
python tools/fill_results.py --root . --card "$CARD" --check
```

If `--check` fails, either the benchmark has not landed or a field in
`results.json` is still null. Do not publish a table with mixed real and
placeholder cells.

## 2. Privacy and attribution gate

Before committing, confirm the tree contains:

- no home directories or machine-specific absolute paths;
- no private-session identifiers or calibration paths;
- no email addresses, API keys, tokens or private keys;
- no attribution or generated-with lines and no co-author trailers.

The preparer ran this gate before every local commit. Re-run it after any edit
to the card or the results. If a hit is a methodological phrase such as
"held-out", check that it carries no path, session text or credential.

## 3. Commit and create the public repository

The repository is already initialized on `main` with local commits. Check what
will be published:

```sh
git status
git log --oneline --decorate
git diff --stat
```

Create the public repository and push the existing history:

```sh
gh repo create boxwrench/Ling-3.1-flash-wrench --public --source . --push
```

Then tag the pages release:

```sh
git tag -a v0 -m "Ling-3.1 Wrench EXL3 2.10 bpw: pages and benchmark harness"
git push origin v0
```

## 4. Upload the model card

The weights are already in `boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw`.
Upload only the reviewed card:

```sh
hf upload boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw \
  "$CARD" README.md \
  --repo-type model \
  --commit-message "Card: one-R9700 benchmark, quality table and reproduction links"
```

Verify the rendered card and the front matter on the Hub. If the card links to
the GitHub repository, confirm the URL resolves before uploading.

## 5. Optional checks

```sh
git ls-remote --heads origin
hf download boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw README.md --local-dir /tmp/card-check
```

Compare the downloaded card with `$CARD`; the only difference should be Hub
rendering.

## Rollback

- GitHub: `gh repo delete boxwrench/Ling-3.1-flash-wrench --yes` (only if the
  repository was just created and contains no wanted history).
- Hugging Face: revert the card commit with `hf upload ... README.md` from the
  previous revision, or use the Hub's commit history to restore it.
- Local: `git reset --hard <previous-commit>` only after confirming the target
  commit and that no wanted work is uncommitted.
