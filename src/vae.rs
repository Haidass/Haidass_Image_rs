//! AutoencoderKLQwenImage21 2D Image Decoder in Rust.
//!
//! Uses channel-last (`HWC`) layout with tiled im2col + AVX-VNNI quantized GEMM for all heavy
//! 3x3 convolutions, keeping `post_quant_conv`, `conv_in`, and `conv_out` in F32 for crisp RGB output.

use crate::format::{HimgFile, QuantMatrixView, TensorDType};
use crate::simd::{axpy_f32, dot_f32, linear, silu_inplace, ActQuantBuf};
use anyhow::Result;
use rayon::prelude::*;

pub const VAE_Z_DIM: usize = 64;

struct ConvLayer<'a> {
    weight: QuantMatrixView<'a>, // [C_out, C_in * kH * kW]
    bias: &'a [f32],
    c_in: usize,
    c_out: usize,
    kernel: usize, // 1 or 3
}

struct ResBlock<'a> {
    norm1_gamma: &'a [f32],
    conv1: ConvLayer<'a>,
    norm2_gamma: &'a [f32],
    conv2: ConvLayer<'a>,
    shortcut: Option<ConvLayer<'a>>,
}

struct MidAttn<'a> {
    norm_gamma: &'a [f32],
    to_qkv: ConvLayer<'a>,
    proj: ConvLayer<'a>,
}

struct UpBlock<'a> {
    resnets: Vec<ResBlock<'a>>,
    upsampler: Option<ConvLayer<'a>>,
    /// (in_c, out_c, factor_t) for exact QwenImage21DupUp3D(first_chunk=True) shortcut
    dup_shortcut: Option<(usize, usize, usize)>,
}

pub struct VaeDecoder<'a> {
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    post_quant_conv: ConvLayer<'a>,
    conv_in: ConvLayer<'a>,
    mid_res0: ResBlock<'a>,
    mid_attn: MidAttn<'a>,
    mid_res1: ResBlock<'a>,
    up_blocks: Vec<UpBlock<'a>>,
    norm_out_gamma: &'a [f32],
    conv_out: ConvLayer<'a>,
}

fn load_conv<'a>(file: &'a HimgFile, name: &str, kernel: usize) -> Result<ConvLayer<'a>> {
    let weight = file.quantized_matrix(&format!("{name}.weight"))?;
    let bias = file.f32_slice(&format!("{name}.bias"))?.1;
    let c_out = weight.n;
    let c_in = weight.k / (kernel * kernel);
    Ok(ConvLayer {
        weight,
        bias,
        c_in,
        c_out,
        kernel,
    })
}

fn load_resblock<'a>(file: &'a HimgFile, prefix: &str) -> Result<ResBlock<'a>> {
    let norm1_gamma = file.f32_slice(&format!("{prefix}.norm1.gamma"))?.1;
    let conv1 = load_conv(file, &format!("{prefix}.conv1"), 3)?;
    let norm2_gamma = file.f32_slice(&format!("{prefix}.norm2.gamma"))?.1;
    let conv2 = load_conv(file, &format!("{prefix}.conv2"), 3)?;
    let shortcut = if file.has(&format!("{prefix}.conv_shortcut.weight")) {
        Some(load_conv(file, &format!("{prefix}.conv_shortcut"), 1)?)
    } else {
        None
    };
    Ok(ResBlock {
        norm1_gamma,
        conv1,
        norm2_gamma,
        conv2,
        shortcut,
    })
}

impl<'a> VaeDecoder<'a> {
    pub fn load(file: &'a HimgFile) -> Result<Self> {
        let latents_mean: Vec<f32> = serde_json::from_value(
            file.header
                .metadata
                .get("vae_latents_mean")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing vae_latents_mean metadata"))?,
        )?;
        let latents_std: Vec<f32> = serde_json::from_value(
            file.header
                .metadata
                .get("vae_latents_std")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing vae_latents_std metadata"))?,
        )?;

        let post_quant_conv = load_conv(file, "vae.post_quant_conv", 1)?;
        let conv_in = load_conv(file, "vae.decoder.conv_in", 3)?;
        let mid_res0 = load_resblock(file, "vae.decoder.mid_block.resnets.0")?;
        let mid_attn = MidAttn {
            norm_gamma: file
                .f32_slice("vae.decoder.mid_block.attentions.0.norm.gamma")?
                .1,
            to_qkv: load_conv(file, "vae.decoder.mid_block.attentions.0.to_qkv", 1)?,
            proj: load_conv(file, "vae.decoder.mid_block.attentions.0.proj", 1)?,
        };
        let mid_res1 = load_resblock(file, "vae.decoder.mid_block.resnets.1")?;

        // dim_mult = [1, 2, 4, 8, 8], decoder_base_dim = 144 -> dims = [1152, 1152, 1152, 576, 288, 144]
        // temperal_downsample = [false, true, true, true] -> temperal_upsample = [true, true, true, false]
        let stage_specs: [(usize, usize, Option<usize>); 5] = [
            (1152, 1152, Some(2)),
            (1152, 1152, Some(2)),
            (1152, 576, Some(2)),
            (576, 288, Some(1)),
            (288, 144, None),
        ];

        let mut up_blocks = Vec::with_capacity(5);
        for (i, &(in_c, out_c, factor_t_opt)) in stage_specs.iter().enumerate() {
            let mut resnets = Vec::with_capacity(3);
            for r in 0..3 {
                resnets.push(load_resblock(
                    file,
                    &format!("vae.decoder.up_blocks.{i}.resnets.{r}"),
                )?);
            }
            let (upsampler, dup_shortcut) = if let Some(ft) = factor_t_opt {
                (
                    Some(load_conv(
                        file,
                        &format!("vae.decoder.up_blocks.{i}.upsampler.resample.1"),
                        3,
                    )?),
                    Some((in_c, out_c, ft)),
                )
            } else {
                (None, None)
            };
            up_blocks.push(UpBlock {
                resnets,
                upsampler,
                dup_shortcut,
            });
        }

        let norm_out_gamma = file.f32_slice("vae.decoder.norm_out.gamma")?.1;
        let conv_out = load_conv(file, "vae.decoder.conv_out", 3)?;

        Ok(Self {
            latents_mean,
            latents_std,
            post_quant_conv,
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out_gamma,
            conv_out,
        })
    }

    /// Decode latent `[64, H_lat, W_lat]` (CHW) into RGB `u8` `[H_out * W_out * 3]`.
    pub fn decode_to_rgb(&self, latent_chw: &[f32], lat_h: usize, lat_w: usize) -> (Vec<u8>, usize, usize) {
        let cells = lat_h * lat_w;
        // Convert CHW -> HWC and denormalize: raw = latent * std + mean
        let mut x = vec![0.0f32; cells * VAE_Z_DIM];
        for idx in 0..cells {
            for c in 0..VAE_Z_DIM {
                x[idx * VAE_Z_DIM + c] =
                    latent_chw[c * cells + idx] * self.latents_std[c] + self.latents_mean[c];
            }
        }

        let mut scratch = ActQuantBuf::default();
        let mut h = lat_h;
        let mut w = lat_w;

        x = run_conv(&x, h, w, &self.post_quant_conv, &mut scratch);
        x = run_conv(&x, h, w, &self.conv_in, &mut scratch);

        // Mid block
        x = run_resblock(&x, h, w, &self.mid_res0, &mut scratch);
        x = run_mid_attn(&x, h, w, &self.mid_attn, &mut scratch);
        x = run_resblock(&x, h, w, &self.mid_res1, &mut scratch);

        // 5 Up blocks
        for up in &self.up_blocks {
            let x_copy = if up.dup_shortcut.is_some() {
                Some(x.clone())
            } else {
                None
            };
            let in_h = h;
            let in_w = w;

            for res in &up.resnets {
                x = run_resblock(&x, h, w, res, &mut scratch);
            }

            if let Some(conv_up) = &up.upsampler {
                let c = conv_up.c_in;
                let x_nn = nearest_upsample_2x(&x, h, w, c);
                h *= 2;
                w *= 2;
                x = run_conv(&x_nn, h, w, conv_up, &mut scratch);
            }

            if let (Some((in_c, out_c, factor_t)), Some(orig)) = (up.dup_shortcut, x_copy) {
                add_dup_up_shortcut(&mut x, &orig, in_h, in_w, in_c, out_c, factor_t);
            }
        }

        // Head: norm_out -> SiLU -> conv_out (144 -> 4)
        rms_norm_hwc(&mut x, self.norm_out_gamma, true);
        let out4 = run_conv(&x, h, w, &self.conv_out, &mut scratch);

        // Extract RGB (channels 0..3), clamp [-1, 1], map to [0, 255]
        let mut rgb = vec![0u8; h * w * 3];
        for i in 0..h * w {
            for c in 0..3 {
                let v = out4[i * 4 + c].clamp(-1.0, 1.0);
                let byte = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + c] = byte;
            }
        }
        (rgb, h, w)
    }
}

/// QwenImage21RMS_norm over channels in HWC layout:
/// `normalized = x / max(||x||_2, 1e-12)`, `out = normalized * sqrt(C) * gamma`, optional fused SiLU.
fn rms_norm_hwc(x: &mut [f32], gamma: &[f32], apply_silu: bool) {
    let c = gamma.len();
    let sqrt_c = (c as f32).sqrt();
    x.par_chunks_mut(c).for_each(|pix| {
        let l2 = dot_f32(pix, pix).sqrt().max(1e-12);
        let s = sqrt_c / l2;
        for j in 0..c {
            pix[j] = pix[j] * s * gamma[j];
        }
        if apply_silu {
            silu_inplace(pix);
        }
    });
}

fn run_resblock(
    x: &[f32],
    h: usize,
    w: usize,
    block: &ResBlock<'_>,
    scratch: &mut ActQuantBuf,
) -> Vec<f32> {
    let h_skip = if let Some(sc) = &block.shortcut {
        run_conv(x, h, w, sc, scratch)
    } else {
        x.to_vec()
    };

    let mut cur = x.to_vec();
    rms_norm_hwc(&mut cur, block.norm1_gamma, true);
    cur = run_conv(&cur, h, w, &block.conv1, scratch);
    rms_norm_hwc(&mut cur, block.norm2_gamma, true);
    let mut out = run_conv(&cur, h, w, &block.conv2, scratch);

    out.par_iter_mut()
        .zip(h_skip.par_iter())
        .for_each(|(o, &s)| *o += s);
    out
}

fn run_mid_attn(
    x: &[f32],
    h: usize,
    w: usize,
    attn: &MidAttn<'_>,
    scratch: &mut ActQuantBuf,
) -> Vec<f32> {
    let n = h * w;
    let c = attn.norm_gamma.len();
    let mut normed = x.to_vec();
    rms_norm_hwc(&mut normed, attn.norm_gamma, false);

    // 1x1 conv -> [n, 3 * c]
    let qkv = run_conv(&normed, h, w, &attn.to_qkv, scratch);
    let scale = 1.0 / (c as f32).sqrt();

    // Single-head SDPA over n = 1024 spatial positions (parallel over query positions)
    let mut ctx = vec![0.0f32; n * c];
    ctx.par_chunks_mut(c).enumerate().for_each(|(qi, out_vec)| {
        let qv = &qkv[qi * 3 * c..qi * 3 * c + c];
        let mut scores = vec![0.0f32; n];
        let mut max_s = f32::NEG_INFINITY;
        for ki in 0..n {
            let kv = &qkv[ki * 3 * c + c..ki * 3 * c + 2 * c];
            let sc = dot_f32(qv, kv) * scale;
            scores[ki] = sc;
            if sc > max_s {
                max_s = sc;
            }
        }
        let mut sum_e = 0.0f32;
        for sc in &mut scores {
            let e = (*sc - max_s).exp();
            *sc = e;
            sum_e += e;
        }
        let inv = 1.0 / sum_e;
        out_vec.fill(0.0);
        for vi in 0..n {
            let vv = &qkv[vi * 3 * c + 2 * c..vi * 3 * c + 3 * c];
            axpy_f32(out_vec, vv, scores[vi] * inv);
        }
    });

    let mut proj = run_conv(&ctx, h, w, &attn.proj, scratch);
    proj.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(p, &orig)| *p += orig);
    proj
}

/// Run 1x1 or 3x3 Conv2d (padding=1 for 3x3) in HWC layout via tiled im2col + SIMD GEMM.
/// Weight K-axis ordering for 3x3 is `(ky, kx, c_in)` to match contiguous HWC slices!
fn run_conv(
    x: &[f32],
    h: usize,
    w: usize,
    conv: &ConvLayer<'_>,
    scratch: &mut ActQuantBuf,
) -> Vec<f32> {
    let c_in = conv.c_in;
    let c_out = conv.c_out;
    let hw = h * w;
    let mut out = vec![0.0f32; hw * c_out];

    if conv.kernel == 1 {
        linear(x, hw, conv.weight, Some(conv.bias), &mut out, scratch);
        return out;
    }

    // 3x3 Conv2d (pad = 1): tile along H so im2col buffer stays compact (~16-32 MB max)
    let k_dim = c_in * 9;
    let target_elems = 4 * 1024 * 1024; // ~16 MB im2col f32 buffer
    let rows_per_strip = (target_elems / (w * k_dim).max(1)).clamp(1, h);
    let mut col_buf = vec![0.0f32; rows_per_strip * w * k_dim];

    let mut y0 = 0;
    while y0 < h {
        let strip_h = (h - y0).min(rows_per_strip);
        let strip_m = strip_h * w;
        let strip_col = &mut col_buf[..strip_m * k_dim];

        // Parallel im2col for rows [y0 .. y0 + strip_h)
        strip_col
            .par_chunks_mut(w * k_dim)
            .enumerate()
            .for_each(|(ry, row_dst)| {
                let y = y0 + ry;
                for x_pos in 0..w {
                    let dst_pix = &mut row_dst[x_pos * k_dim..(x_pos + 1) * k_dim];
                    let mut patch_idx = 0;
                    for dy in -1isize..=1 {
                        let sy = y as isize + dy;
                        for dx in -1isize..=1 {
                            let sx = x_pos as isize + dx;
                            let dst_slot = &mut dst_pix[patch_idx * c_in..(patch_idx + 1) * c_in];
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                let src_off = (sy as usize * w + sx as usize) * c_in;
                                dst_slot.copy_from_slice(&x[src_off..src_off + c_in]);
                            } else {
                                dst_slot.fill(0.0);
                            }
                            patch_idx += 1;
                        }
                    }
                }
            });

        let out_slice = &mut out[y0 * w * c_out..(y0 + strip_h) * w * c_out];
        if conv.weight.dtype == TensorDType::F32 {
            linear(strip_col, strip_m, conv.weight, Some(conv.bias), out_slice, scratch);
        } else {
            linear(strip_col, strip_m, conv.weight, Some(conv.bias), out_slice, scratch);
        }

        y0 += strip_h;
    }

    out
}

fn nearest_upsample_2x(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut out = vec![0.0f32; out_h * out_w * c];
    out.par_chunks_mut(out_w * c)
        .enumerate()
        .for_each(|(oy, dst_row)| {
            let sy = oy / 2;
            let src_row = &x[sy * w * c..(sy + 1) * w * c];
            for ox in 0..out_w {
                let sx = ox / 2;
                dst_row[ox * c..(ox + 1) * c]
                    .copy_from_slice(&src_row[sx * c..(sx + 1) * c]);
            }
        });
    out
}

/// Exact `QwenImage21DupUp3D(first_chunk=True)` shortcut addition for single-frame image (`T=1`).
/// Formula derived from `qwenimage21_vae.py`:
/// `repeats = out_c * factor_t * 4 / in_c`
/// `ic = (((oc * factor_t + (factor_t - 1)) * 2 + dy) * 2 + dx) / repeats`
fn add_dup_up_shortcut(
    out: &mut [f32],
    orig: &[f32],
    _in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    factor_t: usize,
) {
    let out_w = in_w * 2;
    let factor = factor_t * 4;
    let repeats = (out_c * factor) / in_c;
    let t_sub = factor_t - 1;

    // Precompute channel mapping table for each (dy, dx, oc)
    let mut map = [[Vec::with_capacity(out_c), Vec::with_capacity(out_c)], [Vec::with_capacity(out_c), Vec::with_capacity(out_c)]];
    for dy in 0..2 {
        for dx in 0..2 {
            for oc in 0..out_c {
                let flat = ((oc * factor_t + t_sub) * 2 + dy) * 2 + dx;
                map[dy][dx].push(flat / repeats);
            }
        }
    }

    out.par_chunks_mut(out_w * out_c)
        .enumerate()
        .for_each(|(oy, dst_row)| {
            let iy = oy / 2;
            let dy = oy % 2;
            let src_row = &orig[iy * in_w * in_c..(iy + 1) * in_w * in_c];
            for ox in 0..out_w {
                let ix = ox / 2;
                let dx = ox % 2;
                let src_pix = &src_row[ix * in_c..(ix + 1) * in_c];
                let dst_pix = &mut dst_row[ox * out_c..(ox + 1) * out_c];
                let m = &map[dy][dx];
                for oc in 0..out_c {
                    dst_pix[oc] += src_pix[m[oc]];
                }
            }
        });
}
