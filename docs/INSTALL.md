# Install Ling Wrench on Linux

Start by choosing [3.1 EXL3](../ling-3.1/README.md) or [3.0 INT4](../ling-3.0/README.md).
For an agent, use [AI_SETUP](AI_SETUP.md). Commands below distinguish executed verification steps from untested examples.
Do not treat an installation command's success as proof that generation works.

## What you need

| Release | Measured hardware and software | Memory and disk evidence |
|---|---|---|
| 3.1 EXL3 | R9700 32 GB (gfx1201), 9800X3D, 192 GB DDR5; Ubuntu 24.04.5; Python 3.12; ROCm 7.2.1 development tools | 153.6 GB weights; venv 4.5 GiB; JIT cache 54 MiB; inputs 1.3 MiB. Quiet-run allocator VRAM 30.9 GiB, system RAM 133.1 GiB |
| 3.0 INT4 | Radeon 8060S / Ryzen AI Max+ 395; 128 GB unified; Linux/ROCm; vLLM build below | Tested on that memory pool; no measured disk or peak-memory receipt in the published JSON |

Other AMD cards and NVIDIA/CUDA are **untested**. Windows/macOS setup is not covered.
Installed RAM is not available RAM: keep at least 16 GiB available through 3.1 loading and inference, the verification guard.
**System RAM for 3.1:** with the fast settings (62 expert layers on the CPU) peak use was **134.7 GiB system-wide** at 8K and 64K, plus 31.2 GiB of VRAM. The experts are pinned host memory, so the Python process RSS (about 4-5 GB) does not show them: check system-wide available memory instead. 192 GB is tested and leaves about 50 GB free. A 128 GB host does not fit this placement on a 32 GB card and is not a verified 3.1 configuration. Check free disk against the selected HF revision and allow space for dependencies, wheels and caches.

## Ling-3.1 EXL3

### Locations and download

**Untested location/download example.** Run from a directory you own with enough free disk.
These paths are examples; use absolute paths when adapting them. Agree to the weight and dependency downloads first.

```sh
git clone https://github.com/boxwrench/ling-wrench.git
export PUBLIC_REPO="$PWD/ling-wrench"
export EXL3="$PWD/exllamav3"
export MODEL="$PWD/Ling-3.1-flash-exl3-wrench-2.10bpw"
export INPUTS="$PUBLIC_REPO/bench/inputs"
export WORKDIR="$PWD/ling-input-sources"
mkdir -p "$WORKDIR"
```

### Runtime

**Executed setup steps** from the verification command record. The checkout pin resolves to
`0f8cf2c3f052a2503e98c1e77d40b2f055047bcc`.

```sh
git clone https://github.com/turboderp-org/exllamav3.git "$EXL3"
git -C "$EXL3" checkout 0f8cf2c3
python3 -m venv "$EXL3/.venv"
. "$EXL3/.venv/bin/activate"
git -C "$EXL3" apply "$PUBLIC_REPO/bench/exllamav3-rocm-smem.patch"
git -C "$EXL3" apply "$PUBLIC_REPO/bench/exllamav3-explicit-device.patch"
```

Use Python 3.12 for this venv. The exact torch/Triton wheels were installed from local files in verification.
**Untested network-install variant** using those public AMD wheel URLs:

```sh
pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
```

Expected torch import version: `2.9.1+rocm7.2.1.gitff65f5bc`.
The wheel files are 1,650,644,034 bytes (torch) and 287,563,448 bytes (Triton); dependencies need more disk.
A generic torch install gives a different stack. An offline install without cached ordinary dependencies failed.

**Executed dependency/environment steps:**

```sh
pip install -r "$EXL3/requirements.txt"
pip install pyarrow
python -m pip install -r "$EXL3/requirements_examples.txt"
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
export CC=/opt/rocm-7.2.1/llvm/bin/clang CXX=/opt/rocm-7.2.1/llvm/bin/clang++
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
pip install -e "$EXL3"
```

Device 1 was the R9700 on the reference machine. **Untested adaptation:** use device 0 on a single-card PC.
Do not change the device index without checking the GPU name. CUDA spelling in torch's device API also applies to ROCm.
The reference import selected `PYTORCH_ROCM_ARCH=gfx1201`; use the target GPU's architecture on other hardware (untested).

**Untested standalone import variant** of the executed lock-wrapped import:

```sh
python -c 'import exllamav3; import torch; print(torch.__version__)'
```

Wait for exit 0. First import compiles the extension: roughly 3.5–4 minutes was a poll-derived estimate on the reference box.
GCC 13 failed on HIP bf16 headers; the ROCm clang pair above worked.
The [shared-memory patch](../bench/exllamav3-rocm-smem.patch) is required for this torch build's first decode,
not just import. Its 64 KiB ROCm fallback was tested only on gfx1201.

**Untested weight download** (verification used existing local weights):

```sh
hf download boxwrench/Ling-3.1-flash-exl3-wrench-2.10bpw \
  --revision 008db908a26ac55a2999b29617297287b15d187e --local-dir "$MODEL"
```

If `hf` is unavailable, the untested dependency command is:

```sh
pip install huggingface_hub
```

Repeat the same download command to resume. Check errors and remaining disk before retrying.

### Placement and first run

| Status | Hardware | Prompt | CPU-MoE layers `-mcl` | Threads `-mct` | Chunk `-gcs` | Cache `-cs` |
|---|---|---:|---:|---:|---:|---:|
| Measured, fastest (`-dev cuda:0`, `EXL3_MOE_STREAM_T=6`; runner uses `--device cuda:0`) | R9700 32 GB + 192 GB RAM | 8192 and 65536 | 62 | 8 | 8192 | context + 1024 |
| Measured, fastest `examples/chat.py`-equivalent (`model_init` load; console loop not driven) | same | 8192 | 62 | 8 | 8192 | 9216 |
| Measured, `bench/serve_fast.py` (curl: short chat and 8K code) | same | short and 8192 | 62 | 8 | 8192 | 9216 |
| Measured, default autosplit (no `-dev`; also `examples/chat.py` fallback) | same | 8192 | 60 | 8 | 4096 | 9216 |
| Measured, default autosplit | same | 65536 | 62 | 8 | 4096 | 66560 |
| Measured, explicit `-dev` cache sweep | same | 1.5K | 62 | 8 | 8192 | 16384, 20480, 24576, 32768, 40960, 49152, 65536, 66560: all finite at step 0 |
| Untested guidance | smaller GPU / same host | start short | increase from the matching row | start at physical cores | 4096 | leave prompt/output headroom |
| Untested guidance | larger GPU | start short | fewer CPU layers may fit | measure again | 4096 | measure again |

Rows describe one machine, not minimum requirements. More CPU layers free VRAM but need more host RAM.
Thread counts above physical cores reduced decode in the tuning run; do not assume more is faster.

**Cache size and VRAM headroom.** The explicit `-dev` loader places the selected layers and the whole cache on one
card and skips autosplit's VRAM budgeting. On the idle 32 GB R9700 with `-mcl 62`, cache sizes 16384 through 66560 all
produced finite logits at step 0; 66560 reserved 26.54 GiB and left 7.03 GiB free after load (4.60 GiB after a
1.5K-token prompt and 16 new tokens). A 32768-token server that failed at step 0 was running while another process
held 19.4 GiB on the same GPU; the same 32768 command generated 10,213 tokens cleanly once the card was idle. Check
`rocm-smi --showpids` and free VRAM before an explicit load, and do not share the card. If it is shared, use the
default autosplit (it budgets) or a smaller cache, and re-check. Cache size alone was not a limit here.

**Checked flag/mode listing** (does not load the model; exit 0):

```sh
python "$EXL3/examples/chat.py" -m "$MODEL" -mode ling -no_think -temp 0 -mcl 60 -mct 8 -gcs 4096 -cs 9216 -modes
```

Expected: Ling mode is listed. The explicit-device flag and its refusal messages were checked on the pinned checkout.

**Measured interactive path.** This is the `model_init.init` load that `examples/chat.py` runs, with the
explicit-device patch and the public 8K code ids fed non-interactively. It measured **502.4 tok/s prefill /
12.24 tok/s decode** (8192 prompt tokens, greedy 256 generated, 16-token warm-up excluded, two reps:
500.6/12.24 and 504.2/12.24). The benchmark runner measured 505.8/12.20 in the same session. The interactive
console loop itself was not driven.

```sh
EXL3_MOE_STREAM_T=6 python "$EXL3/examples/chat.py" -m "$MODEL" \
  -dev cuda:0 -nw -mode ling -no_think -temp 0 \
  -mcl 62 -mct 8 -gcs 8192 -cs 9216
```

`-nw` skips `model_init`'s post-load warm-up. Without it, the same load alternated about 475 and 500 tok/s across
256-token repetitions; with it, both measured reps were 500–504. The trade is that the first interactive prompt pays
the kernel compile/autotune that the skipped warm-up would have done. `-dev` is refused with `-gs` or `-tp`.
Omit `-dev` and use 60 layers/chunk 4096 for the autosplit fallback; that path still loads and generates.

**Agents and showcases.** [bench/serve_fast.py](../bench/serve_fast.py) loads with the same flags, warms up once,
and serves `POST /v1/chat/completions` on `127.0.0.1` only. It is non-streaming, single-request, uses the Ling chat
template, turns thinking off unless the request sets `"thinking": true`, and is greedy when `temperature` is 0.
Requests may also set `top_p`, `top_k` and `seed`; with thinking on, `reasoning_content` and the token counts are
reported separately and `content` is the answer after `</think>`. Its `timings` field reports the generator's own
prefill/decode seconds and token counts. It was tested with curl on a
short chat (38 prompt / 3 generated, `"pong"`) and the 8K code prompt (8230 prompt / 64 generated: 502.3 tok/s
prefill, 12.06 tok/s decode, 16.38 s TTFT, 0 cached).

```sh
EXL3_MOE_STREAM_T=6 python "$PUBLIC_REPO/bench/serve_fast.py" -m "$MODEL" \
  -dev cuda:0 -nw -mcl 62 -mct 8 -gcs 8192 -cs 9216 --port 8000
```

**Known issue:** with `-nw`, a cold server whose first request was an 8K prompt returned `NonFiniteLogits` at step 0; send one short request (or, in `examples/chat.py`, a short message) before the first long prompt. See [TROUBLESHOOTING](TROUBLESHOOTING.md).

Model loads measured 54–56 seconds with `-nw` (the earlier receipt's 93.8 s included the post-load warm-up).
Files were already local, so this is not a cold-disk or download estimate. Use [REPRODUCE](REPRODUCE.md) for the
executed non-interactive benchmark check.

## Ling-3.0 INT4

Use the same compressed-tensors settings as the official INT4. Measured runtime:
`vLLM 0.26.1rc1.dev467+g838616875`, a ROCm build, with the [full receipt](../ling-3.0/results/runtime.json).
Its install artifact and model revision pins are absent from that JSON; this guide cannot recreate that exact build.
Use an existing compatible installation or consult [vLLM's installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/).
A different build is a new, untested configuration. SGLang/CUDA are untested.

**Untested download and server-launch example**, using the recorded server flags. Set `MODEL` to the chosen local directory.
The generic repo can replace Wrench in the download. The runtime may execute the checkpoint's custom code;
review it before accepting `--trust-remote-code`.

```sh
export MODEL="$PWD/Ling-3.0-flash-int4-wrench"
hf download boxwrench/Ling-3.0-flash-int4-wrench --local-dir "$MODEL"
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --attention-backend TRITON_MLA --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 262144 --gpu-memory-utilization 0.72 --max-num-seqs 6 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
  --mamba-cache-mode align --trust-remote-code
```

This is the unified-memory receipt, not a discrete-GPU offload recipe. Do not promise it fits a single R9700.
Keep the endpoint local. API verification and reporting: [REPRODUCE](REPRODUCE.md#ling-30-serving-check).
