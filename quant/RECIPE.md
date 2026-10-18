# Conversion recipe — Ling-3.1-flash Wrench EXL3 2.10 bpw

This is the exact recipe for the published file. The conversion was run with the
upstream exllamav3 converter on one R9700. The calibration tensor is not
redistributed; its composition is listed below.

## Environment

| Item | Value |
|---|---|
| Converter | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| Measured checkout | `bd4da327` (Ling 3.0/3.1 support, FP8-source fix, `-hq 2`) |
| Required encoder fix | PR [#519](https://github.com/turboderp-org/exllamav3/pull/519), commit `0f8cf2c3` — K = 2 dense-trellis LDS fit on ROCm |
| Pin for reproduction | `0f8cf2c3` or later |
| GPU | AMD Radeon AI PRO R9700, 32 GB, gfx1201 |
| CPU / RAM | AMD Ryzen 7 9800X3D, 192 GB DDR5 |
| OS / ROCm | Linux, ROCm 7.2.1 |
| PyTorch | 2.9.1+rocm7.2.1 |
| Build | clang host compiler (`/opt/rocm-7.2.1/llvm/bin/clang++`), `PYTORCH_ROCM_ARCH=gfx1201` |

## Command

```sh
python convert.py \
  -i "$FP8_SOURCE" \
  -o "$OUT_DIR" \
  -b 2.0 -hb 6 -mb 2 -hq 2 \
  --codebook mul1 \
  --out_scales auto \
  -cd "$WRENCH_CAL" \
  -cr 250 -cc 2048 \
  -d 1
```

- `$FP8_SOURCE` is the Ling-3.1-flash FP8 checkpoint.
- `$OUT_DIR` is the target EXL3 directory (24 shards).
- `$WRENCH_CAL` is a safetensors file with `input_ids` of shape 250 × 2048 =
  512,000 tokens, tokenized with the Ling-3.1 tokenizer.
- `-d 1` keeps the job on one device. `-d 1,0` OOMs on a 20 GB second card
  because the calibrated forward replicates the 512-expert MoE layer.

The strategy is turboderp's `-b 2.0 -hq 2` allocation, module for module: 24
shards, 426,632 index keys, 105,320 tensor-storage entries, 104,448 tensors at
2 bits, 12 at 4 bits, 493 at 6 bits, 367 unquantized; `mul1` codebook; head 6
bits; MTP 2 bits. A byte-level diff against turboderp's 2.10 bpw release shows
identical names, shapes and shard sizes with zero bit mismatches in the strategy
metadata; the only difference is the calibration content.

## Calibration mix

250 sequences × 2,048 tokens, sampled from:

| Share | Content |
|---:|---|
| 45 % | agentic coding and tool-use sessions from our own workflow, excluded from the evaluation corpus |
| 15 % | public coding and agentic traces |
| 10 % | the model's own thinking traces |
| 30 % | generalist floor: English, Chinese, long documents, math and chat |

The session share is described only at this level. The calibration text stays
local and is not part of this repository or the model release. Evaluation
windows are held out before sampling; no evaluation text is used for
calibration.

## Wall time and hardware

The landed run wrote its 24 shards between 2026-10-15 21:44 and 2026-10-16 23:30
local time — about **26 h wall** on one R9700, including an out-of-memory
restart and a resume from layer 4. The patched K = 2 encoder is estimated at
**16–20 h** for this model; the remainder is shard I/O, checkpointing and the
dense layers.

Reference points from the run:

| Stage | Measurement |
|---|---|
| Dense layers 0–3 | 105 / 76 / 76 / 72 s per layer |
| First MoE layer (4) | 512 experts; 16 projections/min (3.7 s/tensor) on the unpatched generic kernel |
| Patched K = 2 tile kernel | 9.5 vs 5.2 Mweights/s; `quantize_exl3` 3.05 → 0.85 s per [4096, 1280] tensor |
| Resume | `-r -d 1`, `next_module_idx = 5` after layer 3 |

The output is 35 files / **153,635,910,264 bytes** (153.6 GB decimal, 143.1 GiB).
All 35 SHA-256 checksums matched the conversion output at landing.

## The RDNA K = 2 encoder fix

On HIP, exllamav3's dense trellis specialization originally ran only for
K = 3/4/5/8. K = 2 needs

```
2 · 16384 · 2 + 256 · 2 + 64 + 128 = 66,240 B
```

(two 32 KB cost arrays, the input tile, padding and the cross-warp argmin),
which is 704 B over RDNA's `sharedMemPerBlockOptin = 65,536`. Every 2-bit tensor
therefore fell back to the generic global-scratch kernel at ~5 Mweights/s,
roughly 74 h for this model. The fix:

1. moves `temp_costs` to shared offset 0;
2. aliases the argmin scratch into that array (the costs are dead by then, with
   a barrier);
3. reads the 256-float input tile from global;
4. selects the dense kernel for K = 2 on ROCm.

The encoding is bit-identical to the generic kernel, the quant kernel suite is
392/392, and the tile kernel is 1.8× faster (`quantize_exl3` about 3.6×). The
change is upstream as PR #519, commit `0f8cf2c3`; it is a conversion speedup,
not an inference speedup.

## Integrity

```sh
sha256sum -c SHA256SUMS   # from the artifact's own manifest directory
```

The published card records the file count, byte count and landing-verification
result. `results/results.json` records the exact model revision used for the
benchmark.
