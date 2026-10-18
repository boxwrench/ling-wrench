# Ling Wrench

Low-bit quants of inclusionAI's Ling models, with public measurements and a reproduction harness.
Weighted toward agentic coding and tool use, with a general-text floor.

## How fast is it?

Prefill means reading the prompt; decode means writing the answer. These are measured results, not estimates.

| Model | Machine | Context / text | Prefill tok/s | Decode tok/s | By / version |
|---|---|---|---:|---:|---|
| 3.1 EXL3 2.10 bpw | R9700 32 GB; 9800X3D; 192 GB RAM | 8K code | 347.6 | 12.21 | maintainer; exllamav3 `0f8cf2c3` + ROCm patch |
| 3.1 EXL3 2.10 bpw | same | 64K code | 348.0 | 7.67 | maintainer; same runtime |

Linux, ROCm 7.2.1; greedy, 256 generated tokens, no speculation or prefix reuse; two runs after a warm-up.
Full table, spread and receipt: [Ling-3.1](ling-3.1/README.md).

| Model | Machine | Prompt | Prefill tok/s | Decode tok/s | By / version |
|---|---|---|---:|---:|---|
| 3.0 INT4 Wrench | 8060S; Max+ 395; 128 GB unified | 8K | 1083.0 | 27.00 | maintainer; vLLM `0.26.1rc1.dev467+g838616875` |
| 3.0 INT4 generic | same | 8K | 1089.2 | 27.74 | maintainer; same runtime |

Linux/ROCm; greedy, 64 generated tokens, MTP enabled; three-run medians, nonce per request.
Ranges and receipt: [Ling-3.0](ling-3.0/README.md). [Add your hardware and version](docs/COMMUNITY_BENCHMARKS.md).

## Which file?

| Model | Format / size | Choose it for | Weights | Page |
|---|---|---|---|---|
| Ling-3.1-flash | EXL3 2.10 bpw; 153.6 GB | GPU plus CPU experts in RAM | [HF](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw) | [3.1](ling-3.1/README.md) |
| Ling-3.0-flash Wrench | compressed-tensors INT4 | vLLM; official INT4 layout; KL to FP8 0.123 vs 0.140 (official) | [HF](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-wrench) | [3.0](ling-3.0/README.md) |
| Ling-3.0-flash generic | compressed-tensors INT4 | alternate rounding, same runtime | [HF](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-generic) | [3.0](ling-3.0/README.md) |

## What you need

| | Tested setup |
|---|---|
| 3.1 | Linux/ROCm; R9700 32 GB; 192 GB host RAM; 153.6 GB weights plus setup space |
| 3.0 | Linux/ROCm; 8060S with 128 GB unified memory; check HF file sizes before downloading |

Other GPUs, CUDA, and smaller RAM sizes are untested. [Requirements and placement](docs/INSTALL.md).

## Install

**Let your AI set it up.** Paste into your coding agent:

```text
Set up Ling Wrench on this PC for me: https://github.com/boxwrench/ling-wrench - follow docs/AI_SETUP.md.
```

**Or do it yourself:** [INSTALL](docs/INSTALL.md).

## Something went wrong?

[TROUBLESHOOTING](docs/TROUBLESHOOTING.md) · [Reproduce](docs/REPRODUCE.md) ·
[Report numbers](docs/COMMUNITY_BENCHMARKS.md) · [Test requests](docs/TEST_REQUESTS.md) ·
[For agents](AGENTS.md) · [llms.txt](llms.txt)

## Licence and credits

Scripts/docs: MIT ([LICENSE](LICENSE)). Base models: inclusionAI (MIT).
Runtime/converter: [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT); serving: vLLM (Apache-2.0).
Benchmark text: PG-19 (DeepMind, Apache-2.0), llama.cpp (MIT). [NOTICE](NOTICE).
The LingWrench engine is published separately.
