# Ling-3.0-flash — Wrench INT4 and generic

Two compressed-tensors S4/g32 quants of [inclusionAI/Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash).
Weights: [Wrench](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-wrench) and
[generic](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-generic).
Both keep the official INT4 tensor layout, configuration, tokenizer and chat template; vLLM serves them without conversion.
Check the selected HF revision's file listing for download size.

## Speed

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

Measured 2026-10-17 on Radeon 8060S / Ryzen AI Max+ 395, 128 GB unified memory,
Linux/ROCm; vLLM `0.26.1rc1.dev467+g838616875`. This is a particular ROCm build, not a reproducible upstream release pin.
Prefill means reading the prompt; decode means writing the answer. This API timing protocol differs from the 3.1 engine timings.
Actual prompt counts are in [runtime.json](results/runtime.json).

<!-- RECEIPT:START -->
| Item | Value |
|---|---|
| Hardware | AMD Radeon 8060S (gfx1151); AMD Ryzen AI Max+ 395; 128 GB unified |
| Software | vLLM 0.26.1rc1.dev467+g838616875 (a ROCm build of vLLM, not an upstream commit; compare checkpoints on one server, not absolute numbers across builds); ROCm; compressed-tensors S4/g32 |
| Method | OpenAI-compatible API; greedy; 64 generated tokens; three runs per prompt length; a random nonce per request, so no prefix reuse. Server flags: `--attention-backend TRITON_MLA --moe-backend triton --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --max-model-len 262144 --gpu-memory-utilization 0.72 --max-num-seqs 6 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align --trust-remote-code`; same flags for all three checkpoints. |
<!-- RECEIPT:END -->

The official same-config range is the comparison floor: keep the full ranges, including the generic short-prompt outlier.
These are serving checks, not evidence of a general speed advantage. MTP is enabled here; it is disabled in the 3.1 table.

## Quality

No task-accuracy result is provided here. Use your own workloads to compare the checkpoints.
The serving table checks loading and generation; it does not measure model quality.

## Run it

[INSTALL](../docs/INSTALL.md#ling-30-int4) contains the vLLM launch and download commands, marked untested where needed.
Use the same server flags for the official INT4, Wrench and generic.
SGLang and CUDA are untested. With a coding agent: [AI_SETUP](../docs/AI_SETUP.md).
Problems: [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md).

## Reproduce

[REPRODUCE](../docs/REPRODUCE.md#ling-30-serving-check) describes the API harness and its output limits.
Receipt and measured rows: [results/runtime.json](results/runtime.json).
The published JSON identifies the runtime build but does not provide its install artifact, model revision pins or measured file sizes.

## Limits

- Measured on one unified-memory Linux/ROCm machine. Discrete-GPU placement and smaller-memory hosts are untested.
- The API harness writes per-server raw output; it does not automatically assemble the combined release JSON.
- No equivalence claim between different vLLM builds, SGLang, or the 3.1 benchmark protocol.

## Credits

inclusionAI for Ling-3.0-flash and the official INT4 (MIT);
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) for serving. [NOTICE](../NOTICE).
