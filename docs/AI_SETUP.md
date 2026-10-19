# Installing Ling Wrench for a user

Instructions for an AI coding agent. Follow steps 0–9 in order and explain progress in plain words.
The human version is [INSTALL](INSTALL.md). Commands there carry tested/untested labels; preserve those labels when advising a user.

## 0. Ground rules

- Inspect before installing. Use a user-owned checkout and venv; ask before changing drivers, system packages or global settings.
- Agree on the release, target disk and download size before fetching weights. The measured 3.1 file is 153.6 GB;
  for 3.0 inspect the selected HF revision's file sizes. Include dependency/cache space and mention metered connections.
- Run long non-interactive work in a logged background job and poll it. Save its PID. Quiet output alone is not failure;
  resume a stopped download with the same command. Stop only the job you started, after agreement with its user.
- Keep servers on localhost. Explain custom-code execution before using the recorded vLLM trust flag.
- Do not launch benchmarks while another process owns the GPU. Never stop another user's work to make room.

## 1. Check the PC

**Untested inspection examples** (read-only; no runtime import or model load):

| Check | Linux command |
|---|---|
| OS / kernel | `cat /etc/os-release; uname -r` |
| CPU / physical cores | `lscpu` |
| RAM / available memory | `free -h` |
| Target disk | `df -h .` |
| AMD GPU and VRAM | `rocm-smi --showproductname --showmeminfo vram` |
| NVIDIA GPU and driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` |
| ROCm development compiler | `/opt/rocm-7.2.1/llvm/bin/clang --version` |

| Path | Evidence | Decision |
|---|---|---|
| Linux, ROCm 7.2.1, R9700 32 GB, 192 GB host RAM | measured for 3.1 (peak about 135 GiB system RAM, 31 GiB VRAM) | use the pinned EXL3 stack; stop if the PC has less than about 150 GB of RAM free |
| Linux/ROCm, 8060S + Max+ 395, 128 GB unified | measured for 3.0 | use a compatible vLLM INT4 build; exact install artifact missing |
| Linux, NVIDIA/CUDA | untested for both | explain the gap; get agreement before trying an upstream CUDA setup |
| Other AMD GPU / less RAM / another OS | untested | report which tested requirement differs; do not promise fit |

If the agreed configuration lacks memory, disk or a working driver/toolkit, explain the missing part and stop setup.
Keep the 3.1 verification guard of at least 16 GiB available RAM throughout loading/inference.

## 2. Get it

Use [INSTALL: locations and download](INSTALL.md#locations-and-download). Ask which Ling release the user wants if unclear.
For 3.1 pin weights to `008db908a26ac55a2999b29617297287b15d187e` and exllamav3 to
`0f8cf2c3f052a2503e98c1e77d40b2f055047bcc` plus the published ROCm patch.
For 3.0 record the actual weight revision and vLLM build; do not substitute a version number for a tested build.

## 3. Pick the configuration

Use the [placement table](INSTALL.md#placement-and-first-run). 3.1 fastest measured: benchmark runner with `--device cuda:0`,
`EXL3_MOE_STREAM_T=6`, 62 CPU-MoE layers, eight threads, chunk 8192. Under autosplit (and `examples/chat.py`): 60 layers at 8K,
62 at 64K, chunk 4096. Smaller/larger-GPU suggestions are untested, not fitted configurations.
For 3.0 use the recorded vLLM settings on the measured unified-memory class only; discrete-GPU placement is untested.

## 4. Install without interactive questions

Follow the chosen section of [INSTALL](INSTALL.md). There is no automatic installer here.
The 3.1 path uses Python 3.12, exact AMD torch/Triton wheels, runtime/example requirements, ROCm clang and the shared-memory patch.
Wait for the extension import to exit successfully before starting generation.
The network-wheel install, weight download and chat generation examples are untested variants of the executed setup.

| Setting | Purpose |
|---|---|
| `CUDA_VISIBLE_DEVICES` | selects the physical GPU before torch starts; index 1 is specific to the reference machine |
| `CC`, `CXX`, `LD_LIBRARY_PATH` | select matching ROCm development tools/libraries |
| `-mcl`, `-mct`, `-gcs`, `-cs` | CPU expert layers, CPU threads, prompt chunk, cache capacity |
| `--host 127.0.0.1` | keeps the vLLM endpoint local |

## 5. While it downloads, tell the user

State the selected file size and download progress. Completion time depends on their connection; no network download time was measured here.
The first extension import may compile for minutes (roughly 3.5–4 minutes was a poll-derived estimate on the R9700 host).
Model loading can make the PC less responsive. Existing local files loaded in about 57–58 seconds in the first matrix;
that is not a cold-start promise. Watch memory rather than assuming a quiet job has hung.
Inference runs locally; setup fetches dependencies and weights from their public hosts.

## 6. Start

For 3.1 use the [flag check and chat launch](INSTALL.md#placement-and-first-run). The chat terminal needs interactive input;
do not claim it exposes an HTTP endpoint. For unattended verification use the benchmark runner in step 7.
For 3.0 use the [vLLM launch](INSTALL.md#ling-30-int4), saving stdout/stderr to a local log.
Background logging and polling are **untested operator examples**; do not overwrite another job's log:

```sh
nohup sh ./start-ling.sh > ling-start.log 2>&1 < /dev/null &
printf '%s\n' "$!" > ling-start.pid
tail -n 30 ling-start.log
```

Create `start-ling.sh` from the selected non-interactive launch in INSTALL first; it is not supplied by this repo.
Have it activate the venv and export the environment. Do not background the interactive chat this way.

## 7. Verify

Follow [REPRODUCE](REPRODUCE.md): build the public inputs, compare against the committed checksums,
then use the executed short 8K shakedown or full matrix. Expect four checksum `OK` lines;
a full matrix needs eight measured outputs of 256 tokens and zero cached tokens/pages.
Generated-id hashes are evidence, not an acceptance target: default greedy runs can differ.
The one-load repeatability screen does not establish portable determinism.

For 3.0, the API check below is **untested here**:

```sh
curl --fail http://127.0.0.1:8000/v1/models
```

Expect a model list containing the served checkpoint; then run the [serving harness](REPRODUCE.md#ling-30-serving-check).
A listening port alone does not prove generation succeeded.

## 8. Connect the user's apps

3.1's documented path is terminal chat and a benchmark, not an API server. Do not invent an endpoint.
For the untested 3.0 launch example the local API base is `http://127.0.0.1:8000/v1`;
use the model id returned by the model-list endpoint. App-specific integrations are untested.
Tell the user the selected model/revision, runtime/build, command, log location and how to stop their own process.

## 9. When something fails

| Symptom | Next step |
|---|---|
| HIP bf16/GCC build error | use the matching ROCm clang pair; wait for a successful import |
| `shared_memory_per_block` missing | apply the published patch at the pinned runtime; gfx1201 fallback is tested |
| VRAM out of memory | check cache/context and other allocations; more CPU layers may help (untested adaptation), but need host RAM |
| Process vanished / RAM pressure | inspect systemd-oomd logs; reduce memory demand; do not disable system protections |
| Non-finite logits with chunk 8192 | chunk 8192 is measured only with `--device cuda:0` and 62 CPU layers; otherwise use 4096 |
| Unexpectedly slow decode | check physical-core thread count and competing transfers/jobs |

More: [TROUBLESHOOTING](TROUBLESHOOTING.md). Give the human the exact command, environment, versions, memory facts,
actual token counts, and log excerpt. Review paths and secrets together before suggesting a [GitHub issue](https://github.com/boxwrench/ling-wrench/issues).
