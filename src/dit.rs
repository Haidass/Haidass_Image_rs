//! PackedSingleStreamDiT (Qwen-Image-2.1 0.6B) Inference Engine in Rust.
//!
//! Optimizations:
//! - Exact Prefix KV-Cache (exploits `causal_condition=True`: text tokens modulate at t=0
//!   and only attend causally to themselves, so all 34 layers' text K/V are computed ONCE).
//! - CFG Batched GEMM (`M = 2048` image tokens across `[cond, uncond]`), sharing weight reads
//!   in L1/L2 cache and quantizing inputs ONCE for fused QKV and fused GateUp.
//! - AVX-VNNI Q8K128 / Q4K128 GEMM with zero per-step heap allocations.
//! - Parallel Flash-style online-softmax SDPA across all 18 heads (9 cond + 9 uncond).
//! - Configurable Euler / Heun / Midpoint ODE samplers, optional TeaCache step skipping,
//!   and optional late-step CFG skipping.

use crate::format::{HimgFile, QuantMatrixView};
use crate::simd::{
    axpy_f32, dot_f32, gated_residual_add, gelu_tanh, layer_norm_modulate, linear, matmul_f32,
    pack_attn_k_q8, pack_attn_v_q8, rms_norm_inplace, sdpa_vnni_qchunk, silu, swiglu_row,
    zero_center_rms_norm, ActQuantBuf,
};
use anyhow::Result;
use rayon::prelude::*;
use std::time::Instant;

pub const DIT_LAYERS: usize = 34;
pub const DIT_HEADS: usize = 9;
pub const DIT_HEAD_DIM: usize = 128;
pub const DIT_DIM: usize = DIT_HEADS * DIT_HEAD_DIM; // 1152
pub const DIT_MLP_DIM: usize = DIT_DIM * 3; // 3456
pub const DIT_IN_CH: usize = 64;
pub const DIT_CTX_DIM: usize = 1024;
pub const DIT_EPS: f32 = 1e-6;
pub const DIT_AXES_ROPE: [usize; 3] = [16, 56, 56];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    Euler,
    Midpoint,
    Heun,
}

#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub steps: usize,
    pub cfg: f32,
    pub timestep_shift: f32,
    pub sampler: Sampler,
    /// Relative L1 change threshold for TeaCache-style residual reuse (0.0 = disabled / exact).
    pub cache_thresh: f32,
    /// Skip CFG (unconditional pass) when `t <= cfg_skip_below` (0.0 = disabled / exact).
    pub cfg_skip_below: f32,
    pub latent_h: usize,
    pub latent_w: usize,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            steps: 50,
            cfg: 8.0,
            timestep_shift: 1.0,
            sampler: Sampler::Euler,
            cache_thresh: 0.0,
            cfg_skip_below: 0.0,
            latent_h: 32,
            latent_w: 32,
        }
    }
}

pub struct SampleStats {
    pub prefill_ms: f64,
    pub denoise_ms: f64,
    pub total_evals: usize,
    pub cached_steps: usize,
    pub cfg_skipped_steps: usize,
}

struct DitBlock<'a> {
    norm_q: &'a [f32],
    norm_k: &'a [f32],
    qkv: QuantMatrixView<'a>,
    to_out: QuantMatrixView<'a>,
    gate_up: QuantMatrixView<'a>,
    mlp_out: QuantMatrixView<'a>,
}

pub struct DitModel<'a> {
    img_in: &'a [f32],          // [1152, 64] F32
    txt_norm: &'a [f32],        // [1024] F32 (with norm_precision folded!)
    txt_in_1: &'a [f32],        // [1152, 1024] F32
    txt_in_2: &'a [f32],        // [1152, 1152] F32
    time_lin1: &'a [f32],       // [1152, 256] F32
    time_lin2: &'a [f32],       // [1152, 1152] F32
    mod_lin: &'a [f32],         // [4608, 1152] F32
    blocks: Vec<DitBlock<'a>>,  // 34 blocks
    norm_out_lin: &'a [f32],    // [1152, 1152] F32
    proj_out: &'a [f32],        // [64, 1152] F32
}

struct PrefixKvCache {
    /// Per layer: K [H, L, 128] and V [H, L, 128]
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    seq_len: usize,
}

struct RopeTable {
    /// [seq_len, 64]
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    fn build_text(text_len: usize) -> Self {
        let half = DIT_HEAD_DIM / 2;
        let inv_freqs = build_rope_inv_freqs();
        let mut cos = vec![0.0f32; text_len * half];
        let mut sin = vec![0.0f32; text_len * half];
        for p in 0..text_len {
            let pf = p as f32;
            let mut off = 0;
            for ax in 0..3 {
                let d_half = DIT_AXES_ROPE[ax] / 2;
                for j in 0..d_half {
                    let ang = pf * inv_freqs[off + j];
                    cos[p * half + off + j] = ang.cos();
                    sin[p * half + off + j] = ang.sin();
                }
                off += d_half;
            }
        }
        Self { cos, sin }
    }

    fn build_image(text_len: usize, h: usize, w: usize) -> Self {
        let half = DIT_HEAD_DIM / 2;
        let inv_freqs = build_rope_inv_freqs();
        let count = h * w;
        let mut cos = vec![0.0f32; count * half];
        let mut sin = vec![0.0f32; count * half];
        let f_pos = text_len as f32;
        let h_start = -((h - h / 2) as isize);
        let w_start = -((w - w / 2) as isize);

        let d0 = DIT_AXES_ROPE[0] / 2; // 8
        let d1 = DIT_AXES_ROPE[1] / 2; // 28
        let d2 = DIT_AXES_ROPE[2] / 2; // 28

        for y in 0..h {
            let hy = (h_start + y as isize) as f32;
            for x in 0..w {
                let wx = (w_start + x as isize) as f32;
                let idx = y * w + x;
                let base = idx * half;
                for j in 0..d0 {
                    let ang = f_pos * inv_freqs[j];
                    cos[base + j] = ang.cos();
                    sin[base + j] = ang.sin();
                }
                for j in 0..d1 {
                    let ang = hy * inv_freqs[d0 + j];
                    cos[base + d0 + j] = ang.cos();
                    sin[base + d0 + j] = ang.sin();
                }
                for j in 0..d2 {
                    let ang = wx * inv_freqs[d0 + d1 + j];
                    cos[base + d0 + d1 + j] = ang.cos();
                    sin[base + d0 + d1 + j] = ang.sin();
                }
            }
        }
        Self { cos, sin }
    }
}

fn build_rope_inv_freqs() -> [f32; DIT_HEAD_DIM / 2] {
    let mut inv = [0.0f32; DIT_HEAD_DIM / 2];
    let mut off = 0;
    for &dim in &DIT_AXES_ROPE {
        let d_half = dim / 2;
        for j in 0..d_half {
            inv[off + j] = 1.0 / (10000.0f64.powf((2 * j) as f64 / (dim as f64))) as f32;
        }
        off += d_half;
    }
    inv
}

#[inline]
fn apply_rope_interleaved(vec: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for j in 0..half {
        let a = vec[2 * j];
        let b = vec[2 * j + 1];
        let c = cos[j];
        let s = sin[j];
        vec[2 * j] = a * c - b * s;
        vec[2 * j + 1] = b * c + a * s;
    }
}

impl<'a> DitModel<'a> {
    /// Load either base (`prefix = "dit"`) or writing-LoRA-merged (`prefix = "dit_writing"`) backbone.
    pub fn load(file: &'a HimgFile, prefix: &str) -> Result<Self> {
        let img_in = file.f32_slice(&format!("{prefix}.img_in"))?.1;
        let txt_norm = file.f32_slice(&format!("{prefix}.txt_in.text_norm"))?.1;
        let txt_in_1 = file.f32_slice(&format!("{prefix}.txt_in.in_layer"))?.1;
        let txt_in_2 = file.f32_slice(&format!("{prefix}.txt_in.out_layer"))?.1;
        let time_lin1 = file.f32_slice(&format!("{prefix}.time_lin1"))?.1;
        let time_lin2 = file.f32_slice(&format!("{prefix}.time_lin2"))?.1;
        let mod_lin = file.f32_slice(&format!("{prefix}.modulation"))?.1;
        let norm_out_lin = file.f32_slice(&format!("{prefix}.norm_out_lin"))?.1;
        let proj_out = file.f32_slice(&format!("{prefix}.proj_out"))?.1;

        let mut blocks = Vec::with_capacity(DIT_LAYERS);
        for l in 0..DIT_LAYERS {
            let bp = format!("{prefix}.blocks.{l}");
            blocks.push(DitBlock {
                norm_q: file.f32_slice(&format!("{bp}.norm_q"))?.1,
                norm_k: file.f32_slice(&format!("{bp}.norm_k"))?.1,
                qkv: file.quantized_matrix(&format!("{bp}.qkv"))?,
                to_out: file.quantized_matrix(&format!("{bp}.to_out"))?,
                gate_up: file.quantized_matrix(&format!("{bp}.gate_up"))?,
                mlp_out: file.quantized_matrix(&format!("{bp}.mlp_out"))?,
            });
        }

        Ok(Self {
            img_in,
            txt_norm,
            txt_in_1,
            txt_in_2,
            time_lin1,
            time_lin2,
            mod_lin,
            blocks,
            norm_out_lin,
            proj_out,
        })
    }

    /// Compute `temb [1152]` and modulation `[scale1, gate1_tanh, scale2, gate2_tanh]` for scalar `t`.
    fn timestep_modulation(&self, t: f32) -> ([f32; DIT_DIM], [f32; DIT_DIM * 4]) {
        let mut proj = [0.0f32; 256];
        let scaled = 1000.0 * t;
        for i in 0..128 {
            let freq = (-(10000.0f64.ln()) * (i as f64) / 128.0).exp() as f32;
            let ang = scaled * freq;
            proj[i] = ang.cos();
            proj[128 + i] = ang.sin();
        }
        let mut h1 = [0.0f32; DIT_DIM];
        matmul_f32(&proj, 1, 256, self.time_lin1, DIT_DIM, None, &mut h1);
        for v in &mut h1 {
            *v = silu(*v);
        }
        let mut temb = [0.0f32; DIT_DIM];
        matmul_f32(&h1, 1, DIT_DIM, self.time_lin2, DIT_DIM, None, &mut temb);

        let mut temb_silu = [0.0f32; DIT_DIM];
        for i in 0..DIT_DIM {
            temb_silu[i] = silu(temb[i]);
        }
        let mut mods = [0.0f32; DIT_DIM * 4];
        matmul_f32(&temb_silu, 1, DIT_DIM, self.mod_lin, DIT_DIM * 4, None, &mut mods);

        // Pre-apply tanh to gate1 (1152..2304) and gate2 (3456..4608)
        for i in DIT_DIM..2 * DIT_DIM {
            mods[i] = mods[i].tanh();
        }
        for i in 3 * DIT_DIM..4 * DIT_DIM {
            mods[i] = mods[i].tanh();
        }
        (temb, mods)
    }

    fn project_text(&self, ctx: &[f32], seq_len: usize) -> Vec<f32> {
        let mut normed = vec![0.0f32; seq_len * DIT_CTX_DIM];
        for t in 0..seq_len {
            zero_center_rms_norm(
                &ctx[t * DIT_CTX_DIM..(t + 1) * DIT_CTX_DIM],
                &mut normed[t * DIT_CTX_DIM..(t + 1) * DIT_CTX_DIM],
                self.txt_norm,
                DIT_EPS,
            );
        }
        let mut h1 = vec![0.0f32; seq_len * DIT_DIM];
        matmul_f32(&normed, seq_len, DIT_CTX_DIM, self.txt_in_1, DIT_DIM, None, &mut h1);
        for v in &mut h1 {
            *v = gelu_tanh(*v);
        }
        let mut out = vec![0.0f32; seq_len * DIT_DIM];
        matmul_f32(&h1, seq_len, DIT_DIM, self.txt_in_2, DIT_DIM, None, &mut out);
        out
    }

    /// Prefill text prefix through all 34 layers at t=0 and capture post-RoPE K, V caches.
    fn prefill_text(&self, ctx: &[f32], seq_len: usize) -> PrefixKvCache {
        let (_, mods0) = self.timestep_modulation(0.0);
        let scale1 = &mods0[0..DIT_DIM];
        let gate1 = &mods0[DIT_DIM..2 * DIT_DIM];
        let scale2 = &mods0[2 * DIT_DIM..3 * DIT_DIM];
        let gate2 = &mods0[3 * DIT_DIM..4 * DIT_DIM];

        let rope = RopeTable::build_text(seq_len);
        let half = DIT_HEAD_DIM / 2;
        let attn_scale = 1.0 / (DIT_HEAD_DIM as f32).sqrt();

        let mut x = self.project_text(ctx, seq_len);
        let mut x_mod = vec![0.0f32; seq_len * DIT_DIM];
        let mut qkv = vec![0.0f32; seq_len * DIT_DIM * 3];
        let mut q_buf = vec![0.0f32; DIT_HEADS * seq_len * DIT_HEAD_DIM];
        let mut attn_out = vec![0.0f32; seq_len * DIT_DIM];
        let mut o_out = vec![0.0f32; seq_len * DIT_DIM];
        let mut gu = vec![0.0f32; seq_len * DIT_MLP_DIM * 2];
        let mut mlp_mid = vec![0.0f32; seq_len * DIT_MLP_DIM];
        let mut scratch = ActQuantBuf::default();

        let mut layers = Vec::with_capacity(DIT_LAYERS);

        for block in &self.blocks {
            // Modulate 1
            for t in 0..seq_len {
                layer_norm_modulate(
                    &x[t * DIT_DIM..(t + 1) * DIT_DIM],
                    &mut x_mod[t * DIT_DIM..(t + 1) * DIT_DIM],
                    scale1,
                    DIT_EPS,
                );
            }
            linear(&x_mod, seq_len, block.qkv, None, &mut qkv, &mut scratch);

            let mut k_cache = vec![0.0f32; DIT_HEADS * seq_len * DIT_HEAD_DIM];
            let mut v_cache = vec![0.0f32; DIT_HEADS * seq_len * DIT_HEAD_DIM];

            for t in 0..seq_len {
                let row = &qkv[t * 3 * DIT_DIM..(t + 1) * 3 * DIT_DIM];
                let cos = &rope.cos[t * half..(t + 1) * half];
                let sin = &rope.sin[t * half..(t + 1) * half];
                for h in 0..DIT_HEADS {
                    let qd = &mut q_buf[(h * seq_len + t) * DIT_HEAD_DIM
                        ..(h * seq_len + t + 1) * DIT_HEAD_DIM];
                    qd.copy_from_slice(&row[h * DIT_HEAD_DIM..(h + 1) * DIT_HEAD_DIM]);
                    rms_norm_inplace(qd, block.norm_q, DIT_EPS);
                    apply_rope_interleaved(qd, cos, sin);

                    let kd = &mut k_cache[(h * seq_len + t) * DIT_HEAD_DIM
                        ..(h * seq_len + t + 1) * DIT_HEAD_DIM];
                    kd.copy_from_slice(
                        &row[DIT_DIM + h * DIT_HEAD_DIM..DIT_DIM + (h + 1) * DIT_HEAD_DIM],
                    );
                    rms_norm_inplace(kd, block.norm_k, DIT_EPS);
                    apply_rope_interleaved(kd, cos, sin);

                    let vd = &mut v_cache[(h * seq_len + t) * DIT_HEAD_DIM
                        ..(h * seq_len + t + 1) * DIT_HEAD_DIM];
                    vd.copy_from_slice(
                        &row[2 * DIT_DIM + h * DIT_HEAD_DIM..2 * DIT_DIM + (h + 1) * DIT_HEAD_DIM],
                    );
                }
            }

            // Causal attention for text prefix
            let mut scores = vec![0.0f32; seq_len];
            let mut acc = [0.0f32; DIT_HEAD_DIM];
            for h in 0..DIT_HEADS {
                let qh = &q_buf[h * seq_len * DIT_HEAD_DIM..(h + 1) * seq_len * DIT_HEAD_DIM];
                let kh = &k_cache[h * seq_len * DIT_HEAD_DIM..(h + 1) * seq_len * DIT_HEAD_DIM];
                let vh = &v_cache[h * seq_len * DIT_HEAD_DIM..(h + 1) * seq_len * DIT_HEAD_DIM];
                for t in 0..seq_len {
                    let qv = &qh[t * DIT_HEAD_DIM..(t + 1) * DIT_HEAD_DIM];
                    let mut max_s = f32::NEG_INFINITY;
                    for s in 0..=t {
                        let sc = dot_f32(qv, &kh[s * DIT_HEAD_DIM..(s + 1) * DIT_HEAD_DIM])
                            * attn_scale;
                        scores[s] = sc;
                        if sc > max_s {
                            max_s = sc;
                        }
                    }
                    let mut sum_e = 0.0f32;
                    for s in 0..=t {
                        let e = (scores[s] - max_s).exp();
                        scores[s] = e;
                        sum_e += e;
                    }
                    let inv = 1.0 / sum_e;
                    acc.fill(0.0);
                    for s in 0..=t {
                        axpy_f32(
                            &mut acc,
                            &vh[s * DIT_HEAD_DIM..(s + 1) * DIT_HEAD_DIM],
                            scores[s] * inv,
                        );
                    }
                    attn_out[t * DIT_DIM + h * DIT_HEAD_DIM..t * DIT_DIM + (h + 1) * DIT_HEAD_DIM]
                        .copy_from_slice(&acc);
                }
            }

            layers.push((k_cache, v_cache));

            linear(&attn_out, seq_len, block.to_out, None, &mut o_out, &mut scratch);
            for t in 0..seq_len {
                let xr = &mut x[t * DIT_DIM..(t + 1) * DIT_DIM];
                let or = &o_out[t * DIT_DIM..(t + 1) * DIT_DIM];
                for j in 0..DIT_DIM {
                    xr[j] += gate1[j] * or[j];
                }
            }

            // Modulate 2 + SwiGLU MLP
            for t in 0..seq_len {
                layer_norm_modulate(
                    &x[t * DIT_DIM..(t + 1) * DIT_DIM],
                    &mut x_mod[t * DIT_DIM..(t + 1) * DIT_DIM],
                    scale2,
                    DIT_EPS,
                );
            }
            linear(&x_mod, seq_len, block.gate_up, None, &mut gu, &mut scratch);
            for t in 0..seq_len {
                let row = &gu[t * 2 * DIT_MLP_DIM..(t + 1) * 2 * DIT_MLP_DIM];
                let mid = &mut mlp_mid[t * DIT_MLP_DIM..(t + 1) * DIT_MLP_DIM];
                for j in 0..DIT_MLP_DIM {
                    mid[j] = silu(row[j]) * row[DIT_MLP_DIM + j];
                }
            }
            linear(&mlp_mid, seq_len, block.mlp_out, None, &mut o_out, &mut scratch);
            for t in 0..seq_len {
                let xr = &mut x[t * DIT_DIM..(t + 1) * DIT_DIM];
                let or = &o_out[t * DIT_DIM..(t + 1) * DIT_DIM];
                for j in 0..DIT_DIM {
                    xr[j] += gate2[j] * or[j];
                }
            }
        }

        PrefixKvCache { layers, seq_len }
    }

    /// Run full denoising loop starting from `latent` `[64, H, W]` (modified in-place).
    pub fn sample(
        &self,
        latent_chw: &mut [f32],
        cond_ctx: &[f32],
        cond_len: usize,
        uncond_ctx: &[f32],
        uncond_len: usize,
        cfg: &SampleConfig,
        mut progress_cb: impl FnMut(usize, usize, f64),
    ) -> SampleStats {
        let cells = cfg.latent_h * cfg.latent_w;
        assert_eq!(latent_chw.len(), DIT_IN_CH * cells);

        // 1. Exact Prefix KV-Cache prefill
        let t0 = Instant::now();
        let kv_cond = self.prefill_text(cond_ctx, cond_len);
        let kv_uncond = self.prefill_text(uncond_ctx, uncond_len);
        let rope_img_c = RopeTable::build_image(cond_len, cfg.latent_h, cfg.latent_w);
        let rope_img_u = RopeTable::build_image(uncond_len, cfg.latent_h, cfg.latent_w);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Build timestep schedule
        let steps = cfg.steps;
        let mut grid = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let mut g = 1.0 - (i as f32) / (steps as f32);
            if cfg.timestep_shift != 1.0 {
                let s = cfg.timestep_shift;
                g = s * g / (1.0 + (s - 1.0) * g);
            }
            grid.push(g);
        }

        // 3. Pre-allocate ALL scratch buffers ONCE (zero heap allocations inside loop)
        let max_m = 2 * cells;
        let mut ws = StepWorkspace::new(max_m, cells);

        let mut total_evals = 0usize;
        let mut cached_steps = 0usize;
        let mut cfg_skipped_steps = 0usize;

        // TeaCache state (relative L1 distance of modulated input to block 0)
        let mut prev_mod_in: Vec<f32> = Vec::new();
        let mut cached_vel: Vec<f32> = Vec::new();
        let mut accum_rel_diff = 0.0f32;

        let mut vel = vec![0.0f32; cells * DIT_IN_CH];
        let mut vel2 = vec![0.0f32; cells * DIT_IN_CH];
        let mut latent_tmp = vec![0.0f32; DIT_IN_CH * cells];

        let t_denoise = Instant::now();
        for step in 0..steps {
            let t_cur = grid[step];
            let t_next = grid[step + 1];
            let dt = t_next - t_cur; // negative
            let step_start = Instant::now();

            let use_cfg = cfg.cfg != 1.0 && !(cfg.cfg_skip_below > 0.0 && t_cur <= cfg.cfg_skip_below);
            if !use_cfg {
                cfg_skipped_steps += 1;
            }

            // Optional TeaCache skip check (only on interior Euler steps)
            let mut reused = false;
            if cfg.cache_thresh > 0.0
                && cfg.sampler == Sampler::Euler
                && step > 1
                && step + 2 < steps
                && !cached_vel.is_empty()
            {
                let rel = self.compute_mod_input_rel_diff(
                    latent_chw,
                    cells,
                    t_cur,
                    &mut ws,
                    &prev_mod_in,
                );
                accum_rel_diff += rel;
                if accum_rel_diff < cfg.cache_thresh {
                    vel.copy_from_slice(&cached_vel);
                    cached_steps += 1;
                    reused = true;
                } else {
                    accum_rel_diff = 0.0;
                }
            }

            if !reused {
                self.eval_velocity(
                    latent_chw,
                    cells,
                    t_cur,
                    cfg.cfg,
                    use_cfg,
                    &kv_cond,
                    &kv_uncond,
                    &rope_img_c,
                    &rope_img_u,
                    &mut ws,
                    &mut vel,
                    if cfg.cache_thresh > 0.0 {
                        Some(&mut prev_mod_in)
                    } else {
                        None
                    },
                );
                total_evals += 1;
                if cfg.cache_thresh > 0.0 {
                    cached_vel.resize(vel.len(), 0.0);
                    cached_vel.copy_from_slice(&vel);
                }
            }

            match cfg.sampler {
                Sampler::Euler => {
                    apply_velocity_chw(latent_chw, &vel, dt, cells);
                }
                Sampler::Midpoint => {
                    let t_mid = 0.5 * (t_cur + t_next);
                    latent_tmp.copy_from_slice(latent_chw);
                    apply_velocity_chw(&mut latent_tmp, &vel, 0.5 * dt, cells);
                    let use_cfg_mid =
                        cfg.cfg != 1.0 && !(cfg.cfg_skip_below > 0.0 && t_mid <= cfg.cfg_skip_below);
                    self.eval_velocity(
                        &latent_tmp,
                        cells,
                        t_mid,
                        cfg.cfg,
                        use_cfg_mid,
                        &kv_cond,
                        &kv_uncond,
                        &rope_img_c,
                        &rope_img_u,
                        &mut ws,
                        &mut vel2,
                        None,
                    );
                    total_evals += 1;
                    apply_velocity_chw(latent_chw, &vel2, dt, cells);
                }
                Sampler::Heun => {
                    if step + 1 == steps {
                        apply_velocity_chw(latent_chw, &vel, dt, cells);
                    } else {
                        latent_tmp.copy_from_slice(latent_chw);
                        apply_velocity_chw(&mut latent_tmp, &vel, dt, cells);
                        let use_cfg_next = cfg.cfg != 1.0
                            && !(cfg.cfg_skip_below > 0.0 && t_next <= cfg.cfg_skip_below);
                        self.eval_velocity(
                            &latent_tmp,
                            cells,
                            t_next,
                            cfg.cfg,
                            use_cfg_next,
                            &kv_cond,
                            &kv_uncond,
                            &rope_img_c,
                            &rope_img_u,
                            &mut ws,
                            &mut vel2,
                            None,
                        );
                        total_evals += 1;
                        for i in 0..vel.len() {
                            vel[i] = 0.5 * (vel[i] + vel2[i]);
                        }
                        apply_velocity_chw(latent_chw, &vel, dt, cells);
                    }
                }
            }

            progress_cb(step + 1, steps, step_start.elapsed().as_secs_f64() * 1000.0);
        }

        SampleStats {
            prefill_ms,
            denoise_ms: t_denoise.elapsed().as_secs_f64() * 1000.0,
            total_evals,
            cached_steps,
            cfg_skipped_steps,
        }
    }

    fn compute_mod_input_rel_diff(
        &self,
        latent_chw: &[f32],
        cells: usize,
        t: f32,
        ws: &mut StepWorkspace,
        prev_mod: &[f32],
    ) -> f32 {
        chw_to_tokens(latent_chw, cells, &mut ws.lat_tok);
        let (_, mods) = self.timestep_modulation(t);
        let scale1 = &mods[0..DIT_DIM];
        matmul_f32(
            &ws.lat_tok,
            cells,
            DIT_IN_CH,
            self.img_in,
            DIT_DIM,
            None,
            &mut ws.x[..cells * DIT_DIM],
        );
        let mut diff = 0.0f64;
        let mut norm = 0.0f64;
        for i in 0..cells {
            layer_norm_modulate(
                &ws.x[i * DIT_DIM..(i + 1) * DIT_DIM],
                &mut ws.x_mod[i * DIT_DIM..(i + 1) * DIT_DIM],
                scale1,
                DIT_EPS,
            );
            let cur = &ws.x_mod[i * DIT_DIM..(i + 1) * DIT_DIM];
            let prv = &prev_mod[i * DIT_DIM..(i + 1) * DIT_DIM];
            for j in 0..DIT_DIM {
                diff += (cur[j] - prv[j]).abs() as f64;
                norm += prv[j].abs() as f64;
            }
        }
        (diff / norm.max(1e-6)) as f32
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_velocity(
        &self,
        latent_chw: &[f32],
        cells: usize,
        t: f32,
        cfg_scale: f32,
        use_cfg: bool,
        kv_cond: &PrefixKvCache,
        kv_uncond: &PrefixKvCache,
        rope_img_c: &RopeTable,
        rope_img_u: &RopeTable,
        ws: &mut StepWorkspace,
        vel_out: &mut [f32],
        mut save_mod_in: Option<&mut Vec<f32>>,
    ) {
        let streams = if use_cfg { 2 } else { 1 };
        let m = streams * cells;

        chw_to_tokens(latent_chw, cells, &mut ws.lat_tok);
        let (temb, mods) = self.timestep_modulation(t);
        let scale1 = &mods[0..DIT_DIM];
        let gate1 = &mods[DIT_DIM..2 * DIT_DIM];
        let scale2 = &mods[2 * DIT_DIM..3 * DIT_DIM];
        let gate2 = &mods[3 * DIT_DIM..4 * DIT_DIM];

        // Project image tokens once, copy to uncond stream if CFG active
        matmul_f32(
            &ws.lat_tok,
            cells,
            DIT_IN_CH,
            self.img_in,
            DIT_DIM,
            None,
            &mut ws.x[..cells * DIT_DIM],
        );
        if use_cfg {
            let (c_part, u_part) = ws.x[..2 * cells * DIT_DIM].split_at_mut(cells * DIT_DIM);
            u_part.copy_from_slice(c_part);
        }

        let x_slice = &mut ws.x[..m * DIT_DIM];
        let x_mod = &mut ws.x_mod[..m * DIT_DIM];
        let qkv = &mut ws.qkv[..m * 3 * DIT_DIM];
        let attn_out = &mut ws.attn_out[..m * DIT_DIM];
        let o_out = &mut ws.o_out[..m * DIT_DIM];
        let gu = &mut ws.gu[..m * 2 * DIT_MLP_DIM];
        let mlp_mid = &mut ws.mlp_mid[..m * DIT_MLP_DIM];

        let half = DIT_HEAD_DIM / 2;
        let attn_scale = 1.0 / (DIT_HEAD_DIM as f32).sqrt();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // 1. Modulate 1 (parallel over tokens)
            x_mod
                .par_chunks_mut(DIT_DIM)
                .enumerate()
                .for_each(|(tok, dst)| {
                    layer_norm_modulate(
                        &x_slice[tok * DIT_DIM..(tok + 1) * DIT_DIM],
                        dst,
                        scale1,
                        DIT_EPS,
                    );
                });

            if layer_idx == 0 {
                if let Some(save) = save_mod_in.as_deref_mut() {
                    save.resize(cells * DIT_DIM, 0.0);
                    save.copy_from_slice(&x_mod[..cells * DIT_DIM]);
                }
            }

            // 2. Fused QKV GEMM
            linear(x_mod, m, block.qkv, None, qkv, &mut ws.act_buf);

            // 3. Extract head-first Q/K/V, apply QK-RMSNorm + Image RoPE (parallel over streams & tokens)
            let q_ptr = ws.q_heads.as_mut_ptr() as usize;
            let k_ptr = ws.k_heads.as_mut_ptr() as usize;
            let v_ptr = ws.v_heads.as_mut_ptr() as usize;

            (0..m).into_par_iter().for_each(|tok| {
                let stream = tok / cells;
                let local_tok = tok % cells;
                let rope = if stream == 0 { rope_img_c } else { rope_img_u };
                let cos = &rope.cos[local_tok * half..(local_tok + 1) * half];
                let sin = &rope.sin[local_tok * half..(local_tok + 1) * half];
                let row = &qkv[tok * 3 * DIT_DIM..(tok + 1) * 3 * DIT_DIM];

                for h in 0..DIT_HEADS {
                    let head_global = stream * DIT_HEADS + h;
                    let off = (head_global * cells + local_tok) * DIT_HEAD_DIM;
                    unsafe {
                        let qd = std::slice::from_raw_parts_mut(
                            (q_ptr as *mut f32).add(off),
                            DIT_HEAD_DIM,
                        );
                        qd.copy_from_slice(&row[h * DIT_HEAD_DIM..(h + 1) * DIT_HEAD_DIM]);
                        rms_norm_inplace(qd, block.norm_q, DIT_EPS);
                        apply_rope_interleaved(qd, cos, sin);

                        let kd = std::slice::from_raw_parts_mut(
                            (k_ptr as *mut f32).add(off),
                            DIT_HEAD_DIM,
                        );
                        kd.copy_from_slice(
                            &row[DIT_DIM + h * DIT_HEAD_DIM..DIT_DIM + (h + 1) * DIT_HEAD_DIM],
                        );
                        rms_norm_inplace(kd, block.norm_k, DIT_EPS);
                        apply_rope_interleaved(kd, cos, sin);

                        let vd = std::slice::from_raw_parts_mut(
                            (v_ptr as *mut f32).add(off),
                            DIT_HEAD_DIM,
                        );
                        vd.copy_from_slice(
                            &row[2 * DIT_DIM + h * DIT_HEAD_DIM
                                ..2 * DIT_DIM + (h + 1) * DIT_HEAD_DIM],
                        );
                    }
                }
            });

            // 4. Parallel AVX-VNNI SageAttention (Q8 QK^T + U8xI8 PV) over all (streams * 9) heads!
            let total_heads = streams * DIT_HEADS;
            // Pack K and V once per head in parallel
            let packed_kv: Vec<(Vec<u8>, Vec<u8>, usize)> = (0..total_heads)
                .into_par_iter()
                .map(|head_global| {
                    let stream = head_global / DIT_HEADS;
                    let h = head_global % DIT_HEADS;
                    let kv = if stream == 0 { kv_cond } else { kv_uncond };
                    let text_len = kv.seq_len;
                    let (k_txt_all, v_txt_all) = &kv.layers[layer_idx];
                    let k_txt = &k_txt_all
                        [h * text_len * DIT_HEAD_DIM..(h + 1) * text_len * DIT_HEAD_DIM];
                    let v_txt = &v_txt_all
                        [h * text_len * DIT_HEAD_DIM..(h + 1) * text_len * DIT_HEAD_DIM];
                    let k_img = &ws.k_heads
                        [head_global * cells * DIT_HEAD_DIM..(head_global + 1) * cells * DIT_HEAD_DIM];
                    let v_img = &ws.v_heads
                        [head_global * cells * DIT_HEAD_DIM..(head_global + 1) * cells * DIT_HEAD_DIM];
                    let kp = pack_attn_k_q8(k_txt, k_img);
                    let vp = pack_attn_v_q8(v_txt, v_img);
                    (kp, vp, text_len + cells)
                })
                .collect();

            let q_tile = 48usize; // multiple of 6 -> hits 6x16 fast path with zero remainder overhead!
            let q_blocks = (cells + q_tile - 1) / q_tile;
            let attn_out_ptr = attn_out.as_mut_ptr() as usize;

            (0..total_heads * q_blocks).into_par_iter().for_each(|task| {
                let head_global = task / q_blocks;
                let qb = task % q_blocks;
                let stream = head_global / DIT_HEADS;
                let h = head_global % DIT_HEADS;
                let q0 = qb * q_tile;
                let q1 = (q0 + q_tile).min(cells);
                let qm = q1 - q0;

                let (ref kp, ref vp, total_kv) = packed_kv[head_global];
                let q_img = &ws.q_heads[(head_global * cells + q0) * DIT_HEAD_DIM
                    ..(head_global * cells + q1) * DIT_HEAD_DIM];
                let mut out_chunk = vec![0.0f32; qm * DIT_HEAD_DIM];
                sdpa_vnni_qchunk(q_img, qm, kp, vp, total_kv, attn_scale, &mut out_chunk);

                unsafe {
                    let base = attn_out_ptr as *mut f32;
                    for r in 0..qm {
                        let tok_idx = stream * cells + q0 + r;
                        let dst = base.add(tok_idx * DIT_DIM + h * DIT_HEAD_DIM);
                        std::ptr::copy_nonoverlapping(
                            out_chunk.as_ptr().add(r * DIT_HEAD_DIM),
                            dst,
                            DIT_HEAD_DIM,
                        );
                    }
                }
            });

            // 5. O projection + gated residual
            linear(attn_out, m, block.to_out, None, o_out, &mut ws.act_buf);
            x_slice
                .par_chunks_mut(DIT_DIM)
                .enumerate()
                .for_each(|(tok, xr)| {
                    gated_residual_add(xr, &o_out[tok * DIT_DIM..(tok + 1) * DIT_DIM], gate1);
                });

            // 6. Modulate 2
            x_mod
                .par_chunks_mut(DIT_DIM)
                .enumerate()
                .for_each(|(tok, dst)| {
                    layer_norm_modulate(
                        &x_slice[tok * DIT_DIM..(tok + 1) * DIT_DIM],
                        dst,
                        scale2,
                        DIT_EPS,
                    );
                });

            // 7. Fused GateUp GEMM + AVX2 SwiGLU
            linear(x_mod, m, block.gate_up, None, gu, &mut ws.act_buf);
            mlp_mid
                .par_chunks_mut(DIT_MLP_DIM)
                .enumerate()
                .for_each(|(tok, mid)| {
                    let row = &gu[tok * 2 * DIT_MLP_DIM..(tok + 1) * 2 * DIT_MLP_DIM];
                    swiglu_row(&row[..DIT_MLP_DIM], &row[DIT_MLP_DIM..], mid);
                });

            // 8. MLP Down GEMM + gated residual
            linear(mlp_mid, m, block.mlp_out, None, o_out, &mut ws.act_buf);
            x_slice
                .par_chunks_mut(DIT_DIM)
                .enumerate()
                .for_each(|(tok, xr)| {
                    gated_residual_add(xr, &o_out[tok * DIT_DIM..(tok + 1) * DIT_DIM], gate2);
                });
        }

        // Final AdaLNContinuous (scale only) + proj_out
        let mut temb_silu = [0.0f32; DIT_DIM];
        for i in 0..DIT_DIM {
            temb_silu[i] = silu(temb[i]);
        }
        let mut scale_out = [0.0f32; DIT_DIM];
        matmul_f32(&temb_silu, 1, DIT_DIM, self.norm_out_lin, DIT_DIM, None, &mut scale_out);

        x_mod
            .par_chunks_mut(DIT_DIM)
            .enumerate()
            .for_each(|(tok, dst)| {
                layer_norm_modulate(
                    &x_slice[tok * DIT_DIM..(tok + 1) * DIT_DIM],
                    dst,
                    &scale_out,
                    DIT_EPS,
                );
            });

        let pred = &mut ws.pred[..m * DIT_IN_CH];
        matmul_f32(x_mod, m, DIT_DIM, self.proj_out, DIT_IN_CH, None, pred);

        if use_cfg {
            let cond = &pred[..cells * DIT_IN_CH];
            let uncond = &pred[cells * DIT_IN_CH..2 * cells * DIT_IN_CH];
            for i in 0..cells * DIT_IN_CH {
                vel_out[i] = uncond[i] + cfg_scale * (cond[i] - uncond[i]);
            }
        } else {
            vel_out.copy_from_slice(&pred[..cells * DIT_IN_CH]);
        }
    }
}

struct StepWorkspace {
    lat_tok: Vec<f32>,
    x: Vec<f32>,
    x_mod: Vec<f32>,
    qkv: Vec<f32>,
    q_heads: Vec<f32>,
    k_heads: Vec<f32>,
    v_heads: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    gu: Vec<f32>,
    mlp_mid: Vec<f32>,
    pred: Vec<f32>,
    act_buf: ActQuantBuf,
}

impl StepWorkspace {
    fn new(max_m: usize, cells: usize) -> Self {
        Self {
            lat_tok: vec![0.0f32; cells * DIT_IN_CH],
            x: vec![0.0f32; max_m * DIT_DIM],
            x_mod: vec![0.0f32; max_m * DIT_DIM],
            qkv: vec![0.0f32; max_m * 3 * DIT_DIM],
            q_heads: vec![0.0f32; 2 * DIT_HEADS * cells * DIT_HEAD_DIM],
            k_heads: vec![0.0f32; 2 * DIT_HEADS * cells * DIT_HEAD_DIM],
            v_heads: vec![0.0f32; 2 * DIT_HEADS * cells * DIT_HEAD_DIM],
            attn_out: vec![0.0f32; max_m * DIT_DIM],
            o_out: vec![0.0f32; max_m * DIT_DIM],
            gu: vec![0.0f32; max_m * 2 * DIT_MLP_DIM],
            mlp_mid: vec![0.0f32; max_m * DIT_MLP_DIM],
            pred: vec![0.0f32; max_m * DIT_IN_CH],
            act_buf: ActQuantBuf::default(),
        }
    }
}

#[inline]
fn chw_to_tokens(chw: &[f32], cells: usize, tokens: &mut [f32]) {
    for idx in 0..cells {
        for c in 0..DIT_IN_CH {
            tokens[idx * DIT_IN_CH + c] = chw[c * cells + idx];
        }
    }
}

#[inline]
fn apply_velocity_chw(latent_chw: &mut [f32], vel_tokens: &[f32], dt: f32, cells: usize) {
    for idx in 0..cells {
        for c in 0..DIT_IN_CH {
            latent_chw[c * cells + idx] += dt * vel_tokens[idx * DIT_IN_CH + c];
        }
    }
}
