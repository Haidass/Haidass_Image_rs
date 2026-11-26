//! Qwen3-0.6B Text Encoder for Haidass-Image-Instruct.
//!
//! 28 Transformer layers, hidden_size=1024, intermediate_size=3072,
//! 16 Q heads, 8 KV heads (GQA), head_dim=128, QK-RMSNorm, RoPE theta=1_000_000.

use crate::format::{HimgFile, QuantMatrixView, TensorDType};
use crate::simd::{axpy_f32, dot_f32, linear, rms_norm_inplace, silu, ActQuantBuf};
use anyhow::{bail, Result};
use rayon::prelude::*;

pub const QWEN3_LAYERS: usize = 28;
pub const QWEN3_HIDDEN: usize = 1024;
pub const QWEN3_INTER: usize = 3072;
pub const QWEN3_Q_HEADS: usize = 16;
pub const QWEN3_KV_HEADS: usize = 8;
pub const QWEN3_HEAD_DIM: usize = 128;
pub const QWEN3_RMS_EPS: f32 = 1e-6;
pub const QWEN3_ROPE_THETA: f32 = 1_000_000.0;
pub const QWEN3_EOS_ID: u32 = 151645;
pub const QWEN3_MAX_LEN: usize = 256;

struct Qwen3Layer<'a> {
    input_ln: &'a [f32],
    qkv: QuantMatrixView<'a>,
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    o_proj: QuantMatrixView<'a>,
    post_ln: &'a [f32],
    gate_up: QuantMatrixView<'a>,
    down_proj: QuantMatrixView<'a>,
}

pub struct Qwen3Encoder<'a> {
    embed_dtype: TensorDType,
    embed_bytes: &'a [u8],
    vocab_size: usize,
    layers: Vec<Qwen3Layer<'a>>,
    final_norm: &'a [f32],
}

impl<'a> Qwen3Encoder<'a> {
    pub fn load(file: &'a HimgFile) -> Result<Self> {
        let (emb_meta, embed_bytes) = file.raw_bytes("qwen3.embed_tokens")?;
        if emb_meta.shape.len() != 2 || emb_meta.shape[1] != QWEN3_HIDDEN {
            bail!("Unexpected qwen3.embed_tokens shape {:?}", emb_meta.shape);
        }
        let vocab_size = emb_meta.shape[0];
        let mut layers = Vec::with_capacity(QWEN3_LAYERS);
        for l in 0..QWEN3_LAYERS {
            let p = format!("qwen3.layers.{l}");
            layers.push(Qwen3Layer {
                input_ln: file.f32_slice(&format!("{p}.input_layernorm"))?.1,
                qkv: file.quantized_matrix(&format!("{p}.qkv"))?,
                q_norm: file.f32_slice(&format!("{p}.q_norm"))?.1,
                k_norm: file.f32_slice(&format!("{p}.k_norm"))?.1,
                o_proj: file.quantized_matrix(&format!("{p}.o_proj"))?,
                post_ln: file.f32_slice(&format!("{p}.post_attention_layernorm"))?.1,
                gate_up: file.quantized_matrix(&format!("{p}.gate_up"))?,
                down_proj: file.quantized_matrix(&format!("{p}.down_proj"))?,
            });
        }
        let final_norm = file.f32_slice("qwen3.norm")?.1;
        Ok(Self {
            embed_dtype: emb_meta.dtype,
            embed_bytes,
            vocab_size,
            layers,
            final_norm,
        })
    }

    fn lookup_embeddings(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let mut x = vec![0.0f32; seq_len * QWEN3_HIDDEN];
        match self.embed_dtype {
            TensorDType::EmbQ8 => {
                let q_len = self.vocab_size * QWEN3_HIDDEN;
                let q_data = &self.embed_bytes[..q_len];
                let scales: &[f32] = bytemuck::cast_slice(&self.embed_bytes[q_len..]);
                for (t, &tid) in token_ids.iter().enumerate() {
                    let id = tid as usize;
                    if id >= self.vocab_size {
                        bail!("Token ID {id} >= vocab_size {}", self.vocab_size);
                    }
                    let s = scales[id];
                    let src = &q_data[id * QWEN3_HIDDEN..(id + 1) * QWEN3_HIDDEN];
                    let dst = &mut x[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
                    for j in 0..QWEN3_HIDDEN {
                        dst[j] = (src[j] as i8 as f32) * s;
                    }
                }
            }
            TensorDType::F32 => {
                let f_data: &[f32] = bytemuck::cast_slice(self.embed_bytes);
                for (t, &tid) in token_ids.iter().enumerate() {
                    let id = tid as usize;
                    if id >= self.vocab_size {
                        bail!("Token ID {id} >= vocab_size {}", self.vocab_size);
                    }
                    x[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN]
                        .copy_from_slice(&f_data[id * QWEN3_HIDDEN..(id + 1) * QWEN3_HIDDEN]);
                }
            }
            other => bail!("Unsupported embedding dtype {:?}", other),
        }
        Ok(x)
    }

    /// Encode a single unpadded sequence of token IDs into `[seq_len, 1024]`.
    pub fn encode(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            bail!("Empty token sequence");
        }
        let mut x = self.lookup_embeddings(token_ids)?;

        // Precompute RoPE cos/sin tables for half-split rotation: [seq_len, 64]
        let half = QWEN3_HEAD_DIM / 2;
        let mut inv_freq = vec![0.0f32; half];
        for i in 0..half {
            inv_freq[i] =
                1.0 / QWEN3_ROPE_THETA.powf((2 * i) as f32 / (QWEN3_HEAD_DIM as f32));
        }
        let mut cos_tab = vec![0.0f32; seq_len * half];
        let mut sin_tab = vec![0.0f32; seq_len * half];
        for pos in 0..seq_len {
            let p = pos as f32;
            for i in 0..half {
                let ang = p * inv_freq[i];
                cos_tab[pos * half + i] = ang.cos();
                sin_tab[pos * half + i] = ang.sin();
            }
        }

        let q_dim = QWEN3_Q_HEADS * QWEN3_HEAD_DIM; // 2048
        let kv_dim = QWEN3_KV_HEADS * QWEN3_HEAD_DIM; // 1024
        let qkv_dim = q_dim + 2 * kv_dim; // 4096

        let mut h_norm = vec![0.0f32; seq_len * QWEN3_HIDDEN];
        let mut qkv = vec![0.0f32; seq_len * qkv_dim];
        let mut q_buf = vec![0.0f32; QWEN3_Q_HEADS * seq_len * QWEN3_HEAD_DIM];
        let mut k_buf = vec![0.0f32; QWEN3_KV_HEADS * seq_len * QWEN3_HEAD_DIM];
        let mut v_buf = vec![0.0f32; QWEN3_KV_HEADS * seq_len * QWEN3_HEAD_DIM];
        let mut attn_ctx = vec![0.0f32; seq_len * q_dim];
        let mut o_out = vec![0.0f32; seq_len * QWEN3_HIDDEN];
        let mut gate_up = vec![0.0f32; seq_len * (2 * QWEN3_INTER)];
        let mut mlp_mid = vec![0.0f32; seq_len * QWEN3_INTER];
        let mut scratch = ActQuantBuf::default();
        let attn_scale = 1.0 / (QWEN3_HEAD_DIM as f32).sqrt();

        for layer in &self.layers {
            // 1. Input RMSNorm
            for t in 0..seq_len {
                let s = &x[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
                let d = &mut h_norm[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
                d.copy_from_slice(s);
                rms_norm_inplace(d, layer.input_ln, QWEN3_RMS_EPS);
            }

            // 2. QKV projection
            linear(&h_norm, seq_len, layer.qkv, None, &mut qkv, &mut scratch);

            // 3. Split Q/K/V into head-first layout, apply QK-RMSNorm + half-split RoPE
            for t in 0..seq_len {
                let row = &qkv[t * qkv_dim..(t + 1) * qkv_dim];
                let cos = &cos_tab[t * half..(t + 1) * half];
                let sin = &sin_tab[t * half..(t + 1) * half];

                for h in 0..QWEN3_Q_HEADS {
                    let dst = &mut q_buf[(h * seq_len + t) * QWEN3_HEAD_DIM
                        ..(h * seq_len + t + 1) * QWEN3_HEAD_DIM];
                    dst.copy_from_slice(&row[h * QWEN3_HEAD_DIM..(h + 1) * QWEN3_HEAD_DIM]);
                    rms_norm_inplace(dst, layer.q_norm, QWEN3_RMS_EPS);
                    apply_rope_half_split(dst, cos, sin);
                }
                for h in 0..QWEN3_KV_HEADS {
                    let k_off = q_dim + h * QWEN3_HEAD_DIM;
                    let v_off = q_dim + kv_dim + h * QWEN3_HEAD_DIM;
                    let k_dst = &mut k_buf[(h * seq_len + t) * QWEN3_HEAD_DIM
                        ..(h * seq_len + t + 1) * QWEN3_HEAD_DIM];
                    k_dst.copy_from_slice(&row[k_off..k_off + QWEN3_HEAD_DIM]);
                    rms_norm_inplace(k_dst, layer.k_norm, QWEN3_RMS_EPS);
                    apply_rope_half_split(k_dst, cos, sin);

                    let v_dst = &mut v_buf[(h * seq_len + t) * QWEN3_HEAD_DIM
                        ..(h * seq_len + t + 1) * QWEN3_HEAD_DIM];
                    v_dst.copy_from_slice(&row[v_off..v_off + QWEN3_HEAD_DIM]);
                }
            }

            // 4. Causal GQA SDPA (parallel across Q heads)
            let attn_ptr = attn_ctx.as_mut_ptr() as usize;
            (0..QWEN3_Q_HEADS).into_par_iter().for_each(|qh| {
                let kvh = qh / (QWEN3_Q_HEADS / QWEN3_KV_HEADS);
                let q_head = &q_buf[qh * seq_len * QWEN3_HEAD_DIM..(qh + 1) * seq_len * QWEN3_HEAD_DIM];
                let k_head = &k_buf[kvh * seq_len * QWEN3_HEAD_DIM..(kvh + 1) * seq_len * QWEN3_HEAD_DIM];
                let v_head = &v_buf[kvh * seq_len * QWEN3_HEAD_DIM..(kvh + 1) * seq_len * QWEN3_HEAD_DIM];
                let mut scores = vec![0.0f32; seq_len];
                let mut out_vec = [0.0f32; QWEN3_HEAD_DIM];
                for t in 0..seq_len {
                    let qv = &q_head[t * QWEN3_HEAD_DIM..(t + 1) * QWEN3_HEAD_DIM];
                    let mut max_s = f32::NEG_INFINITY;
                    for s in 0..=t {
                        let kv = &k_head[s * QWEN3_HEAD_DIM..(s + 1) * QWEN3_HEAD_DIM];
                        let sc = dot_f32(qv, kv) * attn_scale;
                        scores[s] = sc;
                        if sc > max_s {
                            max_s = sc;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for s in 0..=t {
                        let e = (scores[s] - max_s).exp();
                        scores[s] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    out_vec.fill(0.0);
                    for s in 0..=t {
                        let w = scores[s] * inv_sum;
                        let vv = &v_head[s * QWEN3_HEAD_DIM..(s + 1) * QWEN3_HEAD_DIM];
                        axpy_f32(&mut out_vec, vv, w);
                    }
                    unsafe {
                        let dst = (attn_ptr as *mut f32).add(t * q_dim + qh * QWEN3_HEAD_DIM);
                        std::ptr::copy_nonoverlapping(out_vec.as_ptr(), dst, QWEN3_HEAD_DIM);
                    }
                }
            });

            // 5. O projection + residual
            linear(&attn_ctx, seq_len, layer.o_proj, None, &mut o_out, &mut scratch);
            for i in 0..x.len() {
                x[i] += o_out[i];
            }

            // 6. Post-attention RMSNorm + SwiGLU MLP + residual
            for t in 0..seq_len {
                let s = &x[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
                let d = &mut h_norm[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
                d.copy_from_slice(s);
                rms_norm_inplace(d, layer.post_ln, QWEN3_RMS_EPS);
            }
            linear(&h_norm, seq_len, layer.gate_up, None, &mut gate_up, &mut scratch);
            for t in 0..seq_len {
                let gu = &gate_up[t * 2 * QWEN3_INTER..(t + 1) * 2 * QWEN3_INTER];
                let mid = &mut mlp_mid[t * QWEN3_INTER..(t + 1) * QWEN3_INTER];
                for j in 0..QWEN3_INTER {
                    mid[j] = silu(gu[j]) * gu[QWEN3_INTER + j];
                }
            }
            linear(&mlp_mid, seq_len, layer.down_proj, None, &mut o_out, &mut scratch);
            for i in 0..x.len() {
                x[i] += o_out[i];
            }
        }

        // Final RMSNorm
        for t in 0..seq_len {
            let row = &mut x[t * QWEN3_HIDDEN..(t + 1) * QWEN3_HIDDEN];
            rms_norm_inplace(row, self.final_norm, QWEN3_RMS_EPS);
        }

        Ok(x)
    }
}

#[inline]
fn apply_rope_half_split(vec: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let x1 = vec[i];
        let x2 = vec[i + half];
        let c = cos[i];
        let s = sin[i];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
}
