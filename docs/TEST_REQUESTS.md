# Hardware we'd like results from

The measured baselines are [3.1 on R9700](../ling-3.1/README.md) and
[3.0 on 8060S unified memory](../ling-3.0/README.md). Other configurations below are requests, not support promises.
Use [COMMUNITY_BENCHMARKS](COMMUNITY_BENCHMARKS.md); one context or a failed fit check is welcome.

| Wanted (untested) | Why | Current evidence |
|---|---|---|
| NVIDIA 24 / 32 / 48 GB, Linux/CUDA, 3.1 EXL3 | placement, build compatibility and CPU/GPU balance | only R9700/ROCm measured here |
| Other AMD discrete GPUs, 3.1 | check architecture/compiler and shared-memory fallback assumptions | fallback verified only on gfx1201 |
| Hosts with less RAM than the 192 GB 3.1 reference | find fit limits without unsafe memory pressure | no 128 GB host fit claim |
| More memory channels / different CPUs | CPU-expert performance depends on host bandwidth and core count | reference is 9800X3D, eight threads |
| Ling-3.0 with SGLang or CUDA | verify the official INT4 layout on another runtime/backend | vLLM/ROCm receipt only |
| Ling-3.0 on discrete GPUs | document actual placement and required memory | current receipt is 128 GB unified |

Include model revision, full runtime build/patches, prompt/output counts, every repetition, settings, memory and failure logs.
For a comparison, repeat the control and state its spread. Maintainers can close a row when a linked report answers it.
