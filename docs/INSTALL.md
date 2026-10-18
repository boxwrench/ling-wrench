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
A 128 GB host is not a verified 3.1 configuration. Check free disk against the selected HF revision and allow space for dependencies, wheels and caches.

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
| Measured benchmark placement | R9700 32 GB + 192 GB RAM | 8192 | 60 | 8 | 4096 | 9216 |
| Measured benchmark placement | same | 65536 | 62 | 8 | 4096 | 66560 |
| Untested guidance | smaller GPU / same host | start short | increase from the matching row | start at physical cores | 4096 | leave prompt/output headroom |
| Untested guidance | larger GPU | start short | fewer CPU layers may fit | measure again | 4096 | measure again |

Rows describe one machine, not minimum requirements. More CPU layers free VRAM but need more host RAM.
Thread counts above physical cores reduced decode in the tuning run; do not assume more is faster.

**Untested portable flag/mode check**, adapted from the executed lock-wrapped check.
It does not load the model:

```sh
python "$EXL3/examples/chat.py" -m "$MODEL" -mode ling -no_think -temp 0 -mcl 60 -mct 8 -gcs 4096 -cs 9216 -modes
```

Expected: Ling mode is listed, exit 0. For chat, the following generation step is **untested** by the verification job:

```sh
python "$EXL3/examples/chat.py" -m "$MODEL" -mode ling -no_think -temp 0 -mcl 60 -mct 8 -gcs 4096 -cs 9216
```

Keep a chat terminal interactive. Use [REPRODUCE](REPRODUCE.md) for the executed non-interactive generation check.
Model loads measured about 57–58 seconds in the first verified matrix and 93.8 seconds in the quiet receipt;
files were already local, so neither is a cold-disk or download estimate.

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
