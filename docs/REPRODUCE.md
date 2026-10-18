# Reproduce the measurements

The model pages link the measured tables and receipts: [3.1](../ling-3.1/README.md), [3.0](../ling-3.0/README.md).
Keep those runtime/weight versions and all settings when comparing. New hardware is a community measurement, not a reproduction of the same machine.
Install first using [INSTALL](INSTALL.md). The commands here use its `MODEL`, `EXL3`, `PUBLIC_REPO`, `INPUTS` and `WORKDIR` locations.
`PG19` means the local pinned test parquet. Do not run concurrently with GPU work or large host transfers.

## Ling-3.1 inputs

Inputs are one PG-19 test book (DeepMind, Apache-2.0) and sorted llama.cpp `src/*.cpp` (MIT).
The [builder](../bench/build_inputs.py) prepends the runtime BOS and treats special-token-looking text as plain text.
Token ids are not redistributed; [committed checksums](../bench/inputs/SHA256SUMS) are the comparison target.

| Source | Pin |
|---|---|
| Weights/tokenizer | `008db908a26ac55a2999b29617297287b15d187e` |
| PG-19, `emozilla/pg19` test split | `c021754c8e01c5b1cc83a1f549c1f97fbbb756b8` |
| Parquet | `data/test-00000-of-00001-29a571947c0b5ccc.parquet` |
| Parquet SHA-256 | `9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc` |
| llama.cpp | `e467e5b5e5ba43071d6151b3d881deb9300d000e` |

Pins are identifiers, not performance measurements. Verification reused a local parquet; network input fetching is untested.
**Executed source/build commands:** `PG19` must already point at the pinned local parquet.

```sh
git clone https://github.com/ggml-org/llama.cpp "$WORKDIR/llama.cpp"
git -C "$WORKDIR/llama.cpp" checkout e467e5b5e5ba43071d6151b3d881deb9300d000e
python "$PUBLIC_REPO/bench/build_inputs.py" --tokenizer "$MODEL" --pg19 "$PG19" --code-dir "$WORKDIR/llama.cpp" --out "$INPUTS"
```

**Untested network-fetch variant:** use this instead of the local build command if you need the public parquet.
It needs `huggingface_hub` and `pyarrow` (see INSTALL).

```sh
python "$PUBLIC_REPO/bench/build_inputs.py" --tokenizer "$MODEL" \
  --pg19 hf://emozilla/pg19@test --pg19-revision c021754c8e01c5b1cc83a1f549c1f97fbbb756b8 \
  --code-dir "$WORKDIR/llama.cpp" --out "$INPUTS"
```

The builder rewrites `SHA256SUMS`; compare to the committed version, not its newly written copy.
**Untested portable checksum command:**

```sh
git -C "$PUBLIC_REPO" show HEAD:bench/inputs/SHA256SUMS > "$INPUTS/expected.sha256"
(cd "$INPUTS" && sha256sum -c expected.sha256)
```

Expect four `OK` lines. The verification compared all four hashes successfully using a separate comparison script.
The generated manifest can contain local paths; keep it local.

## Ling-3.1 generation

The following **untested portable form of the executed shakedown** uses the smallest recorded generation check: 8K prose and code,
one run each, no warm-up. It still loads the full model and takes substantial memory.
`WORKDIR` is your scratch/output directory.

```sh
python "$PUBLIC_REPO/bench/run_bench.py" --model "$MODEL" --inputs "$INPUTS" --exllamav3 "$EXL3" --contexts 8192 --cpu-moe-layers 60 --threads 8 --chunk 4096 --reps 1 --warmup 0 --out "$WORKDIR/runs/shakedown-8k.json"
```

The enclosed command and settings ran successfully under the reference workstation's lock wrapper.
That external lock is not part of the model setup.

For the speed table, keep the [INSTALL environment](INSTALL.md#runtime), and these **executed environment resets**:

```sh
unset PYTHONPATH EXL3_MOE_STREAM_T EXL3_MOE_STREAM_FUSED_T EXL3_MOE_RECON_DET
```

**Untested output-path variant of the executed full-matrix command**; settings are unchanged, and local paths are represented by variables:

```sh
python "$PUBLIC_REPO/bench/run_bench.py" --model "$MODEL" --inputs "$INPUTS" --exllamav3 "$EXL3" --contexts 8192,65536 --cpu-moe-layers 60,62 --threads 8 --chunk 4096 --reps 2 --warmup 1 --revision 008db908a26ac55a2999b29617297287b15d187e --out "$WORKDIR/runs/results.json"
```

Choose your own output file; do not overwrite the release JSON.
Each context loads separately. Each cell has one excluded 16-token warm-up, then two fresh-generator 256-token runs,
zero cached tokens/pages and no MTP. `--warmup 1` means one pass, not repeated warming to steady placement.
The original verification matrix took 31.17 minutes including imports/builds/loads; that is one measured wall time, not a promise for other hardware.

## Reading the results

The [current speed JSON](../ling-3.1/results/results.json) is the quiet rerun, replacing the earlier transfer-overlapped matrix.
Cell summaries are arithmetic means of two repetitions (with two samples, mean and median coincide).
Relative spread below is `(max - min) / mean`, computed from those repetitions.

| Context / text | Prefill spread | Decode spread |
|---|---:|---:|
| 8K prose | 2.68% | 0.16% |
| 8K code | 0.66% | 1.02% |
| 64K prose | 1.34% | 0.27% |
| 64K code | 0.74% | 1.90% |

R9700 32 GB, 9800X3D, 192 GB RAM; ROCm 7.2.1, exllamav3 `0f8cf2c3` plus the patch.
Do not claim improvements below the larger same-config spread of the compared runs; the release verification also used a minimum 3% gate.
Two repetitions do not establish a precise performance distribution.

Prefill = prompt tokens / engine time to first token; queue time excluded.
Decode = `(generated tokens - 1) / time from first to last token`. Never divide output tokens by total request time.
Retain every repetition, actual generated counts, cached counts, input hashes and generated ids/hashes.
The quiet matrix has eight actual 256-token outputs, zero cached tokens/pages; greedy ids differed across pairs.
Do not use an output hash from one screen as a universal setup test.

Memory fields ending `_gb` are GiB. VRAM is the torch allocator peak; host RAM is MemTotal minus minimum MemAvailable,
not parent RSS. The runner starts host monitoring after loading, so use an external monitor for loading peaks.

## Ling-3.0 serving check

Start the server using [INSTALL](INSTALL.md#ling-30-int4). **Untested portable invocation** of the public API harness
(the 3.1 verification command record contains no executed 3.0 client command):

```sh
python "$PUBLIC_REPO/bench/serve_check.py" http://127.0.0.1:8000/v1 auto "$WORKDIR/serving.json" 3
```

The API needs `/tokenize` and `/detokenize`. The published serving table uses the legacy client formula
`completion_tokens / (last content event time - first content event time)` for decode, which includes the first token
in the numerator. This differs from the 3.1 formula; keep it when reproducing the historical table and label it.
The combined receipt states a 64-token protocol but does not retain per-run actual output counts. The harness sends factual, reasoning, tool-call and short-code checks,
then three runs at about 1K/8K prompt tokens, with a 64-token output limit. Inspect actual `completion_tokens`;
report early stops, errors or missing usage as failures rather than assuming the requested length was generated.
Run each checkpoint with the same flags, including MTP and prefix-cache settings; the request nonce prevents prefix reuse.

The harness writes raw `tasks` and `speed` data. The [combined release JSON](../ling-3.0/results/runtime.json)
contains curated `models[].points[].runs`; assembly is manual and must retain actual counts, failures and build facts.
It does not record model revisions or an install artifact for the measured vLLM build, so exact stack reproduction remains incomplete.
Do not overwrite that combined JSON with one server's raw output.

## Quality

[3.1 quality data](../ling-3.1/results/quality.json) records agreement with the FP8 reference, not task accuracy.
The quality run's machine/runtime and a public end-to-end quality command are not recorded there.
The public harness reproduces speed/serving, not those quality results. Report this limit rather than claiming a full quality reproduction.

## Document checks

These light checks require no model/runtime import. **Executed commands** from verification:

```sh
python3 "$PUBLIC_REPO/tools/test_fill_results.py"
python3 "$PUBLIC_REPO/tools/fill_results.py" --check
```

**Executed during the documentation pass:**

```sh
python3 "$PUBLIC_REPO/tools/check_links.py"
```

To render a local 3.1 card draft, set `CARD` to it. **Executed render/check:**

```sh
python3 "$PUBLIC_REPO/tools/fill_results.py" --profile 3.1 --card "$CARD"
python3 "$PUBLIC_REPO/tools/fill_results.py" --profile 3.1 --check
```

For a 3.0 card, keep its `MODEL_REPO` marker naming the matching checkpoint.
**Executed during the documentation pass** (set `CARD` to either 3.0 draft):

```sh
python3 "$PUBLIC_REPO/tools/fill_results.py" --profile 3.0 --card "$CARD"
```

The renderer's check detects placeholders; it does not prove arbitrary prose or manually added tables match the JSON.
Send results using [COMMUNITY_BENCHMARKS](COMMUNITY_BENCHMARKS.md).
