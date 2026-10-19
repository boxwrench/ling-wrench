# Something went wrong?

Start with the release's [tested requirements](INSTALL.md#what-you-need). Linux/ROCm is measured; CUDA and other hardware are untested.

| Symptom | What happened here | What to do |
|---|---|---|
| HIP bf16 compilation error during first import | GCC 13 failed in the extension's AVX2 source | Use ROCm 7.2.1 clang for both `CC` and `CXX`, as in [INSTALL](INSTALL.md#runtime). A successful editable install is not a successful extension import. |
| First decode: `AttributeError: shared_memory_per_block` | The measured ROCm torch wheel lacks the property read by exllamav3 `0f8cf2c3` | Apply [the patch](../bench/exllamav3-rocm-smem.patch) at that pin. The 64 KiB fallback is tested only on R9700/gfx1201. |
| VRAM out of memory | Cache and placement compete with weights/scratch | Use the matching 8K or 64K placement. Increasing `-mcl` moves more expert layers to host RAM; changing placement on another GPU is untested. Check host headroom first. |
| Process disappears without a kernel OOM message | One verified failed load was killed by systemd-oomd under user-service memory pressure | Read the user/system journal around the failure. Preserve logs and available-memory readings. Do not disable oomd or change global limits to force a fit. |
| RAM short / heavy swapping | 3.1 uses a large CPU expert tier; parent-process RSS misses it | Compare system-wide available memory. 192 GB is tested; 128 GB host RAM is not. Stop your own run if available memory drops below the verification guard of 16 GiB. |
| Non-finite logits at chunk 8192 | Seen under autosplit loading | Use the explicit loader (`--device cuda:0`) with 62 CPU-MoE layers, where chunk 8192 is measured, or use chunk 4096. First rule out the next row: a busy GPU produces the same symptom. Do not publish a throughput from a failed output. |
| Non-finite logits at step 0 with explicit `-dev` | A 32768-token server failed while another process held 19.4 GiB on the same R9700; the same command was finite on an idle card with 7.39 GiB free | The explicit loader skips autosplit's VRAM budget. Check `rocm-smi --showpids` and free VRAM before launch; do not share the card. On an idle 32 GB R9700, cache 16384-66560 all gave finite step-0 logits (66560: 26.54 GiB reserved, 7.03 GiB free). If the card is shared, use autosplit or a smaller cache. |
| More CPU threads make decode slower | Oversubscribing physical cores hurt the tuning run | Start from eight threads on the reference eight-core CPU; measure other CPUs separately. |
| Slow first import or model load | Import compiles; loading reads existing local weights | Wait for progress and monitor memory. Poll-derived import estimate: 3.5–4 minutes. First-matrix loads: about 57–58 seconds; quiet receipt: 93.8 seconds. These are not cold-disk/download timings. |
| Same greedy prompt gives different ids | Default output varied across repetitions | Temperature zero is not a bitwise-output guarantee. Compare valid counts and input hashes; retain output ids. |
| Lower prefill than an earlier table | Warm-up regime and background activity affect CPU-MoE timings | Use the current JSON and exact protocol. Stop competing transfers before measuring; record warm-up passes. Do not attribute a difference below repeat spread. |
| vLLM flags rejected or checkpoint fails to load | The receipt is a particular ROCm build | Record the full build string. A newer release is not a tested replacement. SGLang/CUDA and other builds need their own check. |

The ROCm patch changes the shared-memory-limit lookup, not model weights. Its raw SHA-256 is
`9744c8e95882f25727d9921346611d37f803297eccbdc0d72241f098fa39c0f2`.
The receipt hashes the git diff after stripping its final newline, giving
`e9a50674a7ef9a1f728a75cdc7311ae8a65f954887a5109c3aaa55b4787dc439`.
Those are different hash conventions for the same patch.

The public PyTorch source also excludes the shared-memory property on ROCm; that was source inspection,
not a second-wheel runtime test. Swapping to another torch wheel is untested.
Repeatability controls cost 39.95% more prefill time in the 8K-code screen on this R9700,
so they are not the speed default; the control spread was 0.10% in prefill speed.

## Report a problem

Use [COMMUNITY_BENCHMARKS](COMMUNITY_BENCHMARKS.md), including failures:
GPU/VRAM, CPU/physical cores/RAM, OS/driver, runtime/build and patches, model revision,
all environment variables you set, exact command, logs around the failure, memory at failure,
and actual prompt/generated counts. State whether other jobs were active.
Review logs for local paths and credentials before attaching them to a
[GitHub issue](https://github.com/boxwrench/ling-wrench/issues).
