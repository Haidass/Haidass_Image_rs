# haidass-image-rs

Pure-Rust, zero-Python-runtime CPU inference engine and weight quantizer for [`DALabCommunity/Haidass-Image-Instruct`](https://huggingface.co/DALabCommunity/Haidass-Image-Instruct) (`Qwen-Image-2.1` 0.6B single-stream DiT + `Qwen/Qwen3-0.6B` text encoder + `AutoencoderKLQwenImage21` VAE + on-demand rank-128 Writing LoRA).

## Key Engineering & Optimization Highlights

1. **Custom x86-64 AVX-VNNI (`_mm256_dpbusd_avx_epi32`) `6×16` Quantized GEMM (`Q8K128` / `Q4K128`)**
   - Offline-packed `NR=16, K_group=128` 64-byte aligned weight blocks with precomputed `w_bias` and per-group scales (`Q8K128` cosine similarity vs FP32: **0.999959**, peak **3.27–3.31 TOPS** on Intel Core i7-12700).
   - Online AVX2 dynamic per-128-group activation quantization (`f32 -> u8`), with non-VNNI AVX2 and scalar fallbacks.
   - Fused `QKV` (`[3456, 1152]`) and fused `GateUp` (`[6912, 1152]`) projections cut 7 matmuls/layer down to 4 and quantize activations once per fused projection.

2. **Custom AVX-VNNI SageAttention (`Q8 QK^T` + `U8×I8 PV`)**
   - Uses a dedicated 1-group register-only `6×16` VNNI kernel for `Q @ K^T` (`head_dim=128 == GROUP_SIZE`), AVX2 fast `exp256_ps` Softmax (`max rel err < 2e-7`), and zero-point-0 unsigned `P_u8 @ V_i8^T`.
   - Cuts 18-head CFG attention from `21.9 ms/layer` (AVX2 FP32) down to **`10.16 ms/layer`** (`0.999912` cosine similarity).

3. **Exact Prefix KV-Cache (`causal_condition=True`) + CFG-Batched Execution**
   - Exploits Qwen-Image-2.1's `t=0` causal text conditioning: text tokens run through all 34 blocks **once** (~160 ms), caching post-RoPE `(K_text, V_text)` for all denoising steps with zero accuracy loss.
   - Denoising steps batch conditional & unconditional image streams (`M = 2048`) into a single GEMM pass per projection with zero heap allocations inside the loop.

4. **Zero-Overhead Pre-Merged Writing LoRA + Exact Quote Routing**
   - Exact Rust port of `routing.py`, `add_spelling.py`, `add_spelling_v2.py`, and `spelling.py`.
   - Rank-128 / alpha-256 `writing_lora` is merged in FP32 during conversion (`haidass-q8-writing.himg`), loaded on demand via `mmap` when prompts contain quoted text (`"..."`, `“...”`, `「...」`, etc.) with **0% runtime LoRA overhead**.
   - Folds all 69 `dit_norm_precision.safetensors` FP32 remainders cleanly during conversion.

5. **Stripped & Quantized 2D VAE Decoder + Qwen3-0.6B Text Encoder**
   - Strips unused VAE `encoder`, `quant_conv`, and video `time_conv` weights (`1,288 MB -> ~210 MB`), keeps boundary convs (`post_quant_conv`, `conv_in`, `conv_out`) in FP32, and runs all inner 3×3 convolutions via tiled `HWC` im2col + AVX-VNNI GEMM (~3.8–4.5 s for 512×512).
   - Strips unused Qwen3 `lm_head.weight` (311 MB), embeds `tokenizer.json` directly into the `.himg` file, and encodes prompts in **~115 ms** (vs ~2.4 s in PyTorch CPU).

6. **Optional Fast Sampling Presets**
   - `--preset quality`: Exact 50-step Euler, CFG 8.0, no caching (reference parity).
   - `--preset balanced`: 28-step Euler + light TeaCache (`0.08`) + late CFG skip (`t <= 0.10`) — visually nearly indistinguishable from 50-step quality at **2.6× speedup**.
   - `--preset fast`: 20-step Euler + TeaCache (`0.14`) + late CFG skip (`t <= 0.18`) — **4.4× speedup** (**~17.3 s** total!).

## Benchmarks (Intel Core i7-12700, 12C/20T, AVX-VNNI, `--threads 16`, 512×512)

### Model Size Compression
| Bundle | Original HF Safetensors | `.himg` (`q8`, Recommended) | `.himg` (`q4-mixed`) |
|---|---:|---:|---:|
| Main (Qwen3 + DiT Base + VAE + Tokenizer) | 3,862 MB | **1,503 MB** (2.57× smaller) | **1,013 MB** (3.81× smaller) |
| Writing LoRA (pre-merged DiT) | +383 MB (F32 LoRA) | **636 MB** (0% overhead) | **356 MB** |

### End-to-End 512×512 Generation Latency
| Mode | Quant | Steps (Evals / Cached) | Qwen3 | DiT Prefill | DiT Denoise | VAE Decode | **Total E2E** | Quality Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| `--preset quality` | `q8` | 50 (50 / 0) | 117 ms | 157 ms | 71.65 s | 4.51 s | **76.72 s** | Exact 50-step reference |
| `--preset balanced` | `q8` | 28 (16 / 12) | 120 ms | 177 ms | 23.89 s | 4.60 s | **29.09 s** | Virtually identical to quality |
| `--preset balanced` (Writing LoRA) | `q8` | 28 (16 / 12) | 118 ms | 162 ms | 21.13 s | 3.95 s | **25.65 s** | Crisp auto-routed text |
| `--preset fast` | `q8` | 20 (12 / 8) | 114 ms | 161 ms | 12.95 s | 3.76 s | **17.28 s** | Clean, slightly softer bg |
| `--preset fast` | `q4-mixed` | 20 (12 / 8) | 88 ms | 126 ms | 18.04 s | 4.16 s | **22.71 s** | Slower & degraded on 0.6B DiT |

> **Why `q8` beats `q4` on CPU for this model**: At `M = 2048` (CFG-batched 32×32 latents), DiT linear layers are compute-bound rather than memory-bandwidth-bound. `Q8K128` avoids 4-bit nibble unpacking overhead (3.27 TOPS vs 2.74 TOPS) and preserves near-lossless `0.999959` cosine similarity vs FP32, whereas symmetric 4-bit `Q4K128` degrades fine details on the 0.6B DiT.

## Quick Start

### 1. Build the portable CPU engine (Rust 1.100.0)
```powershell
cargo build --locked --release
```

The Windows x86-64 and Linux GNU x86-64 builds do not use `target-cpu=native`.
The documented timings were measured on one Windows AVX-VNNI machine; they are not
a general minimum-system requirement or a benchmark for other platforms.
Engine archives are independent of model data. See [model provenance and format](docs/model-weights.md)
before downloading or converting weights.

### 2. Convert & Quantize Hugging Face Models (~3.6 seconds)
```powershell
.\target\release\haidass-convert.exe `
  --model-dir models\Haidass-Image-Instruct `
  --text-encoder-dir models\Qwen3-0.6B `
  --quant q8 `
  --output models\haidass-q8.himg `
  --writing-output models\haidass-q8-writing.himg
```

### 3. Generate Images
```powershell
# Recommended balanced mode (~26-29s on CPU)
.\target\release\haidass-infer.exe "A red fox listening at the edge of a frosted field, its tail curled over brown leaves." `
  --model models\haidass-q8.himg --seed 71204 --preset balanced --threads 16 --output fox.png

# Ultra-fast mode (~17s on CPU)
.\target\release\haidass-infer.exe "A red fox listening at the edge of a frosted field, its tail curled over brown leaves." `
  --model models\haidass-q8.himg --seed 71204 --preset fast --threads 16 --output fox_fast.png

# Exact 50-step quality mode (~76s on CPU)
.\target\release\haidass-infer.exe "A red fox listening at the edge of a frosted field, its tail curled over brown leaves." `
  --model models\haidass-q8.himg --seed 71204 --preset quality --threads 16 --output fox_quality.png

# Text rendering (auto-activates spelling + merged writing LoRA)
.\target\release\haidass-infer.exe 'A teal enamel sign reading \"OPEN LATE\" in a rainy shop window' `
  --model models\haidass-q8.himg --seed 41 --preset balanced --threads 16 --output sign.png
```

### 4. Run SIMD Micro-Benchmarks
```powershell
.\target\release\haidass-bench.exe 16
```
