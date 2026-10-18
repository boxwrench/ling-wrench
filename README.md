# Ling Wrench

> **Status, 2026-10-18: early release.** The Ling-3.0 numbers are final. The Ling-3.1 speed numbers were measured on
> 2026-10-17 with our own run script on a locally patched exllamav3 build. Today we re-run them on clean upstream
> exllamav3 with the public harness in `bench/`, add a quality comparison with bartowski's IQ2_XXS GGUF, and add setup
> guides (including one for coding agents). Expect updates here later today.

Wrench is a small family of calibrated low-bit quantizations of inclusionAI's
Ling-3.0-flash and Ling-3.1-flash. Each release keeps the format and launch
settings of the model it is derived from, so it can be used as a drop-in
alternative, and is published with the harness and measurements used to check
it.

The difference between a Wrench release and a generic re-rounding is the
calibration content: Wrench weights the calibration set toward agentic coding
and tool use, with a general-text floor. The release pages report next-token
agreement with the FP8 model on one fixed corpus; those are not task-accuracy
results.

## Releases

| Release | Base model | Format | Size | Weights |
|---|---|---|---:|---|
| Ling-3.1-flash EXL3 2.10 bpw | Ling-3.1-flash | EXL3 trellis | 153.6 GB | [boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw](https://huggingface.co/boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw) |
| Ling-3.0-flash INT4 Wrench | Ling-3.0-flash | compressed-tensors S4/g32 | ~77 GB | [boxwrench/Ling-3.0-flash-int4-wrench](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-wrench) |
| Ling-3.0-flash INT4 generic | Ling-3.0-flash | compressed-tensors S4/g32 | ~77 GB | [boxwrench/Ling-3.0-flash-int4-generic](https://huggingface.co/boxwrench/Ling-3.0-flash-int4-generic) |

The Ling-3.0 INT4 releases are drop-in replacements for
[inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4):
same tensor names, shapes and configuration, with the routed experts re-rounded
from the FP8 weights. The Ling-3.1 release is an EXL3 quant for exllamav3.

## Pages

- [`ling-3.1/`](ling-3.1/) - EXL3 2.10 bpw on one 32 GB GPU: speed, receipt,
  quality and the exact run command.
- [`ling-3.0/`](ling-3.0/) - the two INT4 releases: quality against the official
  INT4, and the vLLM serving check.
- [`bench/`](bench/) - the one harness used for both: 3.1 EXL3 benchmark inputs
  and runner, and the 3.0 OpenAI-API serving check.

The LingWrench engine (quantizer and runtime) will be published separately.

## Licence and credits

The scripts and documentation here are MIT; see [LICENSE](LICENSE). The base
models are MIT-licensed work by inclusionAI. The 3.1 conversion and benchmark use
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) (MIT).
Benchmark inputs are built from PG-19 (DeepMind, Apache-2.0) and llama.cpp (MIT);
token ids are not redistributed. Full notices are in [NOTICE](NOTICE).
