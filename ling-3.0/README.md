# Ling-3.0-flash INT4 — Wrench and generic

Two calibrated INT4 re-roundings of
[inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4),
about **77 GB** each. They keep the official compressed-tensors S4/g32 layout,
tensor names, shapes, configuration, tokenizer and chat template, so they are
drop-in replacements for the official INT4.

- **Wrench:** [boxwrench/Ling-3.0-flash-int4-wrench](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-wrench)
- **generic:** [boxwrench/Ling-3.0-flash-int4-generic](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-generic)

Part of [boxwrench/ling-wrench](https://github.com/boxwrench/ling-wrench).

## Base model

Ling-3.0-flash is inclusionAI's 124B-total, 5.1B-active hybrid MoE: 35 KDA
linear-attention layers and 7 gated MLA layers, 512 routed experts with one
shared expert, and a 256K training context. It is MIT-licensed; see the
[model card](https://huggingface.co/inclusionAI/Ling-3.0-flash).

## This quant

The 61,440 routed-expert projections in the 40 MoE layers were re-rounded from
the FP8 checkpoint with activation-calibrated, GPTQ-style error feedback. All
other tensors are copied byte-for-byte from the official INT4. Wrench used about
1M calibration tokens weighted toward agentic coding and tool use, with a
general-text floor; generic used public corpora only.

## Quality

Full-vocabulary KL and next-token top-1 agreement against
[Ling-3.0-flash-fp8](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp8) on
a locked **92 x 4096-token corpus**. Lower KL and higher top-1 are better.

| Metric | Official INT4 | Wrench | generic |
|---|---:|---:|---:|
| Overall KL ↓ | 0.1402 | **0.1230** | **0.1281** |
| Overall top-1 agreement ↑ | 92.61% | **93.28%** | **93.00%** |
| Agentic sessions KL ↓ | 0.2400 | **0.2093** | **0.2208** |
| Public chat set KL ↓ | 0.0467 | **0.0422** | **0.0411** |
| wikitext-2 KL ↓ | 0.0286 | **0.0261** | **0.0251** |
| Expert-route exact agreement ↑ | 47.58% | **51.55%** | **50.48%** |

Scores measure agreement with the FP8 model on one fixed corpus; they are not
task-accuracy results.

## Serving check

All three checkpoints loaded in vLLM on ROCm with the same format and settings.
Greedy factual, reasoning, tool-call and short-code checks matched the official
run. Median throughput is within a few percent of the official INT4; the official
three-run spread is 0.2-0.3% prefill and 9.4-12.5% decode. One generic 1K
prefill run was 387 tok/s, visible in the range.

<!-- RUNTIME:START -->
| Model | Status | Prompt | Prefill tok/s | Decode tok/s | TTFT (s) |
|---|---|---:|---:|---:|---:|
| Official INT4 | SERVES OK | 1K | 960.1 [958.9-961.9] | 29.56 [27.02-29.57] | 1.09 [1.09-1.09] |
| Official INT4 | SERVES OK | 8K | 1081.6 [1081.0-1082.9] | 26.17 [25.44-28.62] | 7.59 [7.59-7.60] |
| Wrench | SERVES OK | 1K | 961.4 [949.6-962.1] | 27.86 [27.36-29.70] | 1.09 [1.09-1.10] |
| Wrench | SERVES OK | 8K | 1083.0 [891.1-1084.2] | 27.00 [26.99-28.70] | 7.58 [7.58-9.22] |
| Generic | SERVES OK | 1K | 953.4 [386.7-956.8] | 29.49 [28.11-29.54] | 1.10 [1.09-2.71] |
| Generic | SERVES OK | 8K | 1089.2 [1083.6-1089.5] | 27.74 [25.43-28.07] | 7.54 [7.54-7.58] |

Median of the runs; brackets are min-max. OpenAI-compatible API; greedy; 64 generated tokens; three runs per prompt length; a random nonce per request, so no prefix reuse. Server flags: `--attention-backend TRITON_MLA --moe-backend triton --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --max-model-len 262144 --gpu-memory-utilization 0.72 --max-num-seqs 6 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align --trust-remote-code`; same flags for all three checkpoints.
<!-- RUNTIME:END -->

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon 8060S (gfx1151); AMD Ryzen AI Max+ 395; 128 GB unified |
| Software | vLLM 0.26.1rc1.dev467+g838616875 (a ROCm build of vLLM, not an upstream commit; compare checkpoints on one server, not absolute numbers across builds); ROCm; compressed-tensors S4/g32 |
| Method | OpenAI-compatible API; greedy; 64 generated tokens; three runs per prompt length; a random nonce per request, so no prefix reuse. Server flags: `--attention-backend TRITON_MLA --moe-backend triton --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --max-model-len 262144 --gpu-memory-utilization 0.72 --max-num-seqs 6 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align --trust-remote-code`; same flags for all three checkpoints. |
<!-- RECEIPT:END -->

## Run it

Use the same runtime and launch settings as the official INT4. For SGLang, use
inclusionAI's Ling fork and the official card's server command. The checkpoints
are also served by vLLM with its compressed-tensors INT4 path; no config change
or conversion is needed.

```sh
hf download boxwrench/Ling-3.0-flash-int4-wrench --local-dir ./Ling-3.0-flash-int4-wrench
```

Point `--model-path` / `--model` at that directory. The generic checkpoint uses
the same command.

## Reproduce the serving check

`bench/serve_check.py` drives any OpenAI-compatible endpoint. It sends the four
greedy sanity prompts, then measures prompt and decode throughput at about 1K
and 8K tokens with three runs per point.

```sh
python bench/serve_check.py http://127.0.0.1:8000/v1 MODEL OUTPUT.json 3
```

`bench/README.md` has the full protocol. Results are in
`ling-3.0/results/runtime.json`.

## Credits and license

- Base model and official INT4: [inclusionAI](https://huggingface.co/inclusionAI), MIT.
  The upstream notice is reproduced in [NOTICE](../NOTICE).
- Re-rounding: boxwrench. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
