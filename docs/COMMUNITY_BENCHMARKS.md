# Report your benchmark

One model, GPU or context is enough. Failed runs and setups that do not fit are useful too.
Start from [REPRODUCE](REPRODUCE.md); wanted hardware is listed in [TEST_REQUESTS](TEST_REQUESTS.md).

## What to record

- GPU model/count, VRAM, CPU and physical cores; installed/available RAM, memory speed/channels, PCIe link and storage.
- OS/kernel, driver/toolkit, torch version, complete runtime build/commit and local patches.
- Model HF repo and exact revision; format, size, tokenizer revision and input hashes.
- Exact command and all environment variables you set; context/cache/placement, threads, chunk, sampling and MTP.
- Warm-up passes, prefix-cache state, prompt and actual generated counts, every repetition, TTFT and separate prefill/decode.
- Peak-memory source/units and whether loading was monitored; background jobs, power limits, temperatures and failures.

Do not fill unknown facts with guesses. Mark them unknown, and label estimates as estimates.

## How to measure

Keep other GPU jobs and large host transfers out of the run. Use the same inputs/configuration for comparisons.
The maintainer's 3.1 release used two repetitions after a warm-up; community runs should preferably use at least three,
retaining every result and reporting median plus min–max. Label single runs explicitly.
Report actual output length; an early stop is not the requested token count.

Prefill reads the prompt. Decode writes the answer: exclude the first token and its time.
The 3.1 TTFT is engine prefill time; the 3.0 API client includes request/stream overhead and its historical decode
formula includes the first token in the numerator ([details](REPRODUCE.md#ling-30-serving-check)).
State which clock you used. Do not mix these protocols in a speedup claim.
Record prefix reuse and MTP even if disabled. Keep same-config repeat spread beside any improvement/regression;
compare against the larger spread of both arms. Describe a timing confound instead of explaining it away.

## Report template: the existing results JSON

Use [ling-3.1/results/results.json](../ling-3.1/results/results.json) as the complete template,
not a second incompatible format. It uses schema `ling-r9700-bench/1`, kind `exl3-bench`:

| Fields | Put here |
|---|---|
| `date`, `model` | measurement date; HF repo/revision, format size |
| `hardware` | machine and full software/runtime receipt |
| `config` | placement, threads, chunk, environment, exact command |
| `inputs` | source, actual token length, SHA-256 per window |
| `cells[].reps` | every repetition: speeds, timings, actual/cached token counts, output-id hash/ids |
| `cells` | context/domain and summary with the aggregation method stated |
| `peak_vram_gb`, `peak_ram_gb` | legacy names: GiB; say which monitor and whether loading is included |

Replace the example's machine/results with yours; never retain its measurements as defaults.
For 3.0 use [ling-3.0/results/runtime.json](../ling-3.0/results/runtime.json),
same schema with kind `serving-check`, and attach the raw API harness output too.
Its `models[].points[].runs` holds repetitions per checkpoint/prompt size.
Keep failure logs separately and explain missing cells. The renderer is a table formatter, not a schema validator.

## Share

Open a [GitHub issue](https://github.com/boxwrench/ling-wrench/issues) or a pull request with the JSON,
a short readable report and a log excerpt. Include your handle and runtime version for a front-page row.
State what you did not test. Review commands, manifests and logs for personal paths, credentials, private prompts and addresses first.
An agent should show its human the report before suggesting publication.
