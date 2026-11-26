//! High-performance x86-64 AVX-VNNI / AVX2 quantized & F32 compute kernels.
//!
//! Core GEMM computes `C[M, N] = A[M, K] @ B[N, K]^T` (+ optional bias)
//! where activations `A` are dynamically quantized per 128-element K-group to `u8` (shifted by +128)
//! and weights `B` are offline-packed into `NR=16, K_group=128` tiles (`Q8K128` or `Q4K128`).

use crate::format::{
    align_up, QuantMatrixView, TensorDType, GROUP_SIZE, NR_TILE, Q4K128_BLOCK_BYTES,
    Q8K128_BLOCK_BYTES,
};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Quantize a weight matrix `[N, K]` (row-major f32) into packed `Q8K128` layout.
/// Output bytes length = `(N_pad / 16) * (K_pad / 128) * Q8K128_BLOCK_BYTES`.
pub fn pack_q8k128(weights: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(weights.len(), n * k);
    let n_pad = align_up(n, NR_TILE);
    let k_pad = align_up(k, GROUP_SIZE);
    let n_tiles = n_pad / NR_TILE;
    let k_groups = k_pad / GROUP_SIZE;

    let mut out = vec![0u8; n_tiles * k_groups * Q8K128_BLOCK_BYTES];

    out.par_chunks_mut(k_groups * Q8K128_BLOCK_BYTES)
        .enumerate()
        .for_each(|(nt, tile_buf)| {
            let n0 = nt * NR_TILE;
            for g in 0..k_groups {
                let k0 = g * GROUP_SIZE;
                let block = &mut tile_buf[g * Q8K128_BLOCK_BYTES..(g + 1) * Q8K128_BLOCK_BYTES];
                let (w_bytes, rest) = block.split_at_mut(2048);
                let (bias_bytes, scale_bytes) = rest.split_at_mut(64);
                let biases: &mut [i32] = bytemuck::cast_slice_mut(bias_bytes);
                let scales: &mut [f32] = bytemuck::cast_slice_mut(scale_bytes);

                for c in 0..NR_TILE {
                    let row_idx = n0 + c;
                    if row_idx >= n {
                        scales[c] = 0.0;
                        biases[c] = 0;
                        continue;
                    }
                    let row = &weights[row_idx * k..(row_idx + 1) * k];
                    let k_end = (k0 + GROUP_SIZE).min(k);
                    let mut max_abs = 0.0f32;
                    for &v in &row[k0..k_end] {
                        let a = v.abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }
                    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    scales[c] = scale;

                    let mut sum_q = 0i32;
                    let half_off = if c < 8 { c * 4 } else { 32 + (c - 8) * 4 };
                    for k4 in 0..32 {
                        let base = k4 * 64 + half_off;
                        for r in 0..4 {
                            let ki = k0 + k4 * 4 + r;
                            let q = if ki < k {
                                (row[ki] * inv_scale).round().clamp(-127.0, 127.0) as i8
                            } else {
                                0i8
                            };
                            sum_q += q as i32;
                            w_bytes[base + r] = q as u8;
                        }
                    }
                    biases[c] = sum_q * 128;
                }
            }
        });

    out
}

/// Quantize a weight matrix `[N, K]` (row-major f32) into packed `Q4K128` layout.
/// Each weight is quantized symmetrically to `[-7, 7]`, stored as nibble `q + 8` in `[1, 15]`.
/// Two consecutive K4 steps (`2*p` and `2*p+1`) share 64 bytes: low nibble = `k4=2*p`, high nibble = `k4=2*p+1`.
pub fn pack_q4k128(weights: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(weights.len(), n * k);
    let n_pad = align_up(n, NR_TILE);
    let k_pad = align_up(k, GROUP_SIZE);
    let n_tiles = n_pad / NR_TILE;
    let k_groups = k_pad / GROUP_SIZE;

    let mut out = vec![0u8; n_tiles * k_groups * Q4K128_BLOCK_BYTES];

    out.par_chunks_mut(k_groups * Q4K128_BLOCK_BYTES)
        .enumerate()
        .for_each(|(nt, tile_buf)| {
            let n0 = nt * NR_TILE;
            for g in 0..k_groups {
                let k0 = g * GROUP_SIZE;
                let block = &mut tile_buf[g * Q4K128_BLOCK_BYTES..(g + 1) * Q4K128_BLOCK_BYTES];
                let (w_bytes, rest) = block.split_at_mut(1024);
                let (bias_bytes, scale_bytes) = rest.split_at_mut(64);
                let biases: &mut [i32] = bytemuck::cast_slice_mut(bias_bytes);
                let scales: &mut [f32] = bytemuck::cast_slice_mut(scale_bytes);

                // Initialize nibbles to 8 (zero weight) so padded entries decode to 0
                w_bytes.fill(0x88);

                for c in 0..NR_TILE {
                    let row_idx = n0 + c;
                    if row_idx >= n {
                        scales[c] = 0.0;
                        biases[c] = 0;
                        continue;
                    }
                    let row = &weights[row_idx * k..(row_idx + 1) * k];
                    let k_end = (k0 + GROUP_SIZE).min(k);
                    let mut max_abs = 0.0f32;
                    for &v in &row[k0..k_end] {
                        let a = v.abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }
                    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
                    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    scales[c] = scale;

                    let mut sum_q = 0i32;
                    let half_off = if c < 8 { c * 4 } else { 32 + (c - 8) * 4 };
                    for p in 0..16 {
                        let base = p * 64 + half_off;
                        for r in 0..4 {
                            let ki_lo = k0 + (2 * p) * 4 + r;
                            let ki_hi = k0 + (2 * p + 1) * 4 + r;
                            let q_lo = if ki_lo < k {
                                (row[ki_lo] * inv_scale).round().clamp(-7.0, 7.0) as i8
                            } else {
                                0i8
                            };
                            let q_hi = if ki_hi < k {
                                (row[ki_hi] * inv_scale).round().clamp(-7.0, 7.0) as i8
                            } else {
                                0i8
                            };
                            sum_q += q_lo as i32 + q_hi as i32;
                            let u_lo = (q_lo + 8) as u8;
                            let u_hi = (q_hi + 8) as u8;
                            w_bytes[base + r] = (u_lo & 0x0F) | ((u_hi & 0x0F) << 4);
                        }
                    }
                    biases[c] = sum_q * 128;
                }
            }
        });

    out
}

/// Pack embedding table `[vocab, dim]` into row-wise INT8 (`EmbQ8`).
pub fn pack_emb_q8(weights: &[f32], vocab: usize, dim: usize) -> Vec<u8> {
    assert_eq!(weights.len(), vocab * dim);
    let mut out = vec![0u8; vocab * dim + vocab * 4];
    let (q_part, s_part) = out.split_at_mut(vocab * dim);
    let scales: &mut [f32] = bytemuck::cast_slice_mut(s_part);

    q_part
        .par_chunks_mut(dim)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(r, (qrow, scale_out))| {
            let row = &weights[r * dim..(r + 1) * dim];
            let mut max_abs = 0.0f32;
            for &v in row {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            *scale_out = scale;
            for j in 0..dim {
                qrow[j] = (row[j] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
            }
        });
    out
}

/// Scratch buffer for dynamically quantized activations so GEMM does zero allocations inside loops.
#[derive(Default)]
pub struct ActQuantBuf {
    pub q_u8: Vec<u8>,
    pub scales: Vec<f32>,
    pub m: usize,
    pub k_pad: usize,
    pub k_groups: usize,
}

impl ActQuantBuf {
    pub fn quantize(&mut self, a: &[f32], m: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        let k_pad = align_up(k, GROUP_SIZE);
        let k_groups = k_pad / GROUP_SIZE;
        self.m = m;
        self.k_pad = k_pad;
        self.k_groups = k_groups;
        self.q_u8.resize(m * k_pad, 128u8);
        self.scales.resize(m * k_groups, 0.0);

        let quant_row = |row_idx: usize, qrow: &mut [u8], srow: &mut [f32]| {
            let src = &a[row_idx * k..(row_idx + 1) * k];
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    unsafe {
                        quantize_row_avx2(src, qrow, srow, k, k_groups);
                    }
                    return;
                }
            }
            quantize_row_scalar(src, qrow, srow, k, k_groups);
        };

        if m >= 32 {
            self.q_u8
                .par_chunks_mut(k_pad)
                .zip(self.scales.par_chunks_mut(k_groups))
                .enumerate()
                .for_each(|(r, (qrow, srow))| quant_row(r, qrow, srow));
        } else {
            for r in 0..m {
                let qrow = &mut self.q_u8[r * k_pad..(r + 1) * k_pad];
                let srow = &mut self.scales[r * k_groups..(r + 1) * k_groups];
                quant_row(r, qrow, srow);
            }
        }
    }
}

#[inline]
fn quantize_row_scalar(src: &[f32], qrow: &mut [u8], srow: &mut [f32], k: usize, k_groups: usize) {
    for g in 0..k_groups {
        let k0 = g * GROUP_SIZE;
        let k_end = (k0 + GROUP_SIZE).min(k);
        let mut max_abs = 0.0f32;
        for &v in &src[k0..k_end] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        srow[g] = scale;
        for j in 0..GROUP_SIZE {
            let idx = k0 + j;
            if idx < k {
                let q = (src[idx] * inv).round().clamp(-127.0, 127.0) as i32;
                qrow[idx] = (q + 128) as u8;
            } else {
                qrow[idx] = 128;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_avx2(
    src: &[f32],
    qrow: &mut [u8],
    srow: &mut [f32],
    k: usize,
    k_groups: usize,
) {
    unsafe {
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
        let v_min = _mm256_set1_ps(-127.0);
        let v_max = _mm256_set1_ps(127.0);
        let v_128 = _mm256_set1_epi32(128);
        let perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        for g in 0..k_groups {
            let k0 = g * GROUP_SIZE;
            if k0 + GROUP_SIZE <= k {
                let sp = src.as_ptr().add(k0);
                let mut vmax0 = _mm256_setzero_ps();
                let mut vmax1 = _mm256_setzero_ps();
                let mut j = 0;
                while j < GROUP_SIZE {
                    let v0 = _mm256_and_ps(_mm256_loadu_ps(sp.add(j)), abs_mask);
                    let v1 = _mm256_and_ps(_mm256_loadu_ps(sp.add(j + 8)), abs_mask);
                    vmax0 = _mm256_max_ps(vmax0, v0);
                    vmax1 = _mm256_max_ps(vmax1, v1);
                    j += 16;
                }
                let vmax = _mm256_max_ps(vmax0, vmax1);
                let hi = _mm256_extractf128_ps(vmax, 1);
                let lo = _mm256_castps256_ps128(vmax);
                let m128 = _mm_max_ps(lo, hi);
                let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
                let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
                let max_abs = _mm_cvtss_f32(m32);

                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                *srow.get_unchecked_mut(g) = scale;

                let vinv = _mm256_set1_ps(inv);
                let dp = qrow.as_mut_ptr().add(k0);
                let mut j = 0;
                while j < GROUP_SIZE {
                    // Process 32 floats -> 32 u8 bytes
                    let f0 = _mm256_min_ps(
                        _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(sp.add(j)), vinv), v_min),
                        v_max,
                    );
                    let f1 = _mm256_min_ps(
                        _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 8)), vinv), v_min),
                        v_max,
                    );
                    let f2 = _mm256_min_ps(
                        _mm256_max_ps(
                            _mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 16)), vinv),
                            v_min,
                        ),
                        v_max,
                    );
                    let f3 = _mm256_min_ps(
                        _mm256_max_ps(
                            _mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 24)), vinv),
                            v_min,
                        ),
                        v_max,
                    );

                    let i0 = _mm256_add_epi32(_mm256_cvtps_epi32(f0), v_128);
                    let i1 = _mm256_add_epi32(_mm256_cvtps_epi32(f1), v_128);
                    let i2 = _mm256_add_epi32(_mm256_cvtps_epi32(f2), v_128);
                    let i3 = _mm256_add_epi32(_mm256_cvtps_epi32(f3), v_128);

                    // Pack 32-bit -> 16-bit -> 8-bit, then fix AVX2 lane interleaving!
                    let p01 = _mm256_packus_epi32(i0, i1); // lane0: 0..3,8..11 | lane1: 4..7,12..15
                    let p23 = _mm256_packus_epi32(i2, i3); // lane0: 16..19,24..27 | lane1: 20..23,28..31
                    let p0123 = _mm256_packus_epi16(p01, p23);
                    // Now 32-bit dwords are: [0..3, 8..11, 16..19, 24..27, 4..7, 12..15, 20..23, 28..31]
                    let ordered = _mm256_permutevar8x32_epi32(p0123, perm_idx);
                    _mm256_storeu_si256(dp.add(j) as *mut __m256i, ordered);
                    j += 32;
                }
            } else {
                // Tail group
                let k_end = k;
                let mut max_abs = 0.0f32;
                for &v in &src[k0..k_end] {
                    let a = v.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                *srow.get_unchecked_mut(g) = scale;
                for j in 0..GROUP_SIZE {
                    let idx = k0 + j;
                    if idx < k {
                        let q = (src[idx] * inv).round().clamp(-127.0, 127.0) as i32;
                        *qrow.get_unchecked_mut(idx) = (q + 128) as u8;
                    } else {
                        *qrow.get_unchecked_mut(idx) = 128;
                    }
                }
            }
        }
    }
}

/// Dispatch GEMM: `out[M, N] = act_q @ w^T (+ optional bias)`.
/// Supports `Q8K128`, `Q4K128`, and `F32` (when called via `linear_f32_or_quant`).
pub fn matmul_quant(
    act: &ActQuantBuf,
    w: QuantMatrixView<'_>,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let m = act.m;
    let n = w.n;
    assert_eq!(out.len(), m * n);
    let n_pad = align_up(n, NR_TILE);
    let n_tiles = n_pad / NR_TILE;
    let k_groups = act.k_groups;
    assert_eq!(k_groups, align_up(w.k, GROUP_SIZE) / GROUP_SIZE);

    match w.dtype {
        TensorDType::Q8K128 => {
            matmul_q8k128_parallel(act, w.data, n, n_tiles, k_groups, bias, out);
        }
        TensorDType::Q4K128 => {
            matmul_q4k128_parallel(act, w.data, n, n_tiles, k_groups, bias, out);
        }
        other => panic!("matmul_quant does not support {:?}", other),
    }
}

/// Convenience: quantize `a` and run quantized matmul, or run F32 matmul if weight is F32.
pub fn linear(
    a: &[f32],
    m: usize,
    w: QuantMatrixView<'_>,
    bias: Option<&[f32]>,
    out: &mut [f32],
    scratch: &mut ActQuantBuf,
) {
    if w.dtype == TensorDType::F32 {
        let wf: &[f32] = bytemuck::cast_slice(w.data);
        matmul_f32(a, m, w.k, wf, w.n, bias, out);
    } else {
        scratch.quantize(a, m, w.k);
        matmul_quant(scratch, w, bias, out);
    }
}

/// Parallel tiled Q8K128 GEMM.
fn matmul_q8k128_parallel(
    act: &ActQuantBuf,
    w_data: &[u8],
    n: usize,
    n_tiles: usize,
    k_groups: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let m = act.m;
    let k_pad = act.k_pad;
    let tile_stride = k_groups * Q8K128_BLOCK_BYTES;

    #[cfg(target_arch = "x86_64")]
    let use_vnni = std::arch::is_x86_feature_detected!("avxvnni");
    #[cfg(not(target_arch = "x86_64"))]
    let use_vnni = false;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    // Choose MC block size (multiple of 6) so work distributes finely across P-cores & E-cores
    let mc = if m >= 1024 {
        258
    } else if m >= 256 {
        126
    } else {
        m.max(1)
    };
    let m_blocks = (m + mc - 1) / mc;
    let total_tasks = m_blocks * n_tiles;

    let out_ptr = out.as_mut_ptr() as usize;
    let a_ptr = act.q_u8.as_ptr() as usize;
    let s_ptr = act.scales.as_ptr() as usize;
    let w_ptr = w_data.as_ptr() as usize;
    let b_ptr = bias.map(|b| b.as_ptr() as usize).unwrap_or(0);

    let run_task = |task_id: usize| {
        let mb = task_id / n_tiles;
        let nt = task_id % n_tiles;
        let m0 = mb * mc;
        let m1 = (m0 + mc).min(m);
        let n0 = nt * NR_TILE;
        let n_valid = (n - n0).min(NR_TILE);
        let w_tile = unsafe { (w_ptr as *const u8).add(nt * tile_stride) };
        let a_base = a_ptr as *const u8;
        let s_base = s_ptr as *const f32;
        let o_base = out_ptr as *mut f32;
        let bias_ptr = if b_ptr != 0 {
            unsafe { Some((b_ptr as *const f32).add(n0)) }
        } else {
            None
        };

        let mut mr_cur = m0;
        while mr_cur + 6 <= m1 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_vnni {
                    microkernel_q8_6x16_vnni(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                } else if use_avx2 {
                    microkernel_q8_6x16_avx2(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                } else {
                    microkernel_q8_scalar(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        6,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe {
                microkernel_q8_scalar(
                    a_base.add(mr_cur * k_pad),
                    s_base.add(mr_cur * k_groups),
                    w_tile,
                    o_base.add(mr_cur * n + n0),
                    bias_ptr,
                    6,
                    k_pad,
                    k_groups,
                    n,
                    n_valid,
                );
            }
            mr_cur += 6;
        }
        let rem = m1 - mr_cur;
        if rem > 0 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_vnni {
                    microkernel_q8_remx16_vnni(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        rem,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                } else {
                    microkernel_q8_scalar(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        rem,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe {
                microkernel_q8_scalar(
                    a_base.add(mr_cur * k_pad),
                    s_base.add(mr_cur * k_groups),
                    w_tile,
                    o_base.add(mr_cur * n + n0),
                    bias_ptr,
                    rem,
                    k_pad,
                    k_groups,
                    n,
                    n_valid,
                );
            }
        }
    };

    if total_tasks >= 8 && m * n * k_groups >= 65536 {
        (0..total_tasks).into_par_iter().for_each(run_task);
    } else {
        for t in 0..total_tasks {
            run_task(t);
        }
    }
}

/// Core 6x16 AVX-VNNI Q8K128 microkernel: 12 YMM i32 accumulators, zero spills in K-loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn microkernel_q8_6x16_vnni(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];

        let a0_base = a_ptr;
        let a1_base = a_ptr.add(k_pad);
        let a2_base = a_ptr.add(k_pad * 2);
        let a3_base = a_ptr.add(k_pad * 3);
        let a4_base = a_ptr.add(k_pad * 4);
        let a5_base = a_ptr.add(k_pad * 5);

        for g in 0..k_groups {
            let block = w_tile.add(g * Q8K128_BLOCK_BYTES);
            let k_off = g * GROUP_SIZE;

            let a0 = a0_base.add(k_off) as *const i32;
            let a1 = a1_base.add(k_off) as *const i32;
            let a2 = a2_base.add(k_off) as *const i32;
            let a3 = a3_base.add(k_off) as *const i32;
            let a4 = a4_base.add(k_off) as *const i32;
            let a5 = a5_base.add(k_off) as *const i32;

            let mut c00 = _mm256_setzero_si256();
            let mut c01 = _mm256_setzero_si256();
            let mut c10 = _mm256_setzero_si256();
            let mut c11 = _mm256_setzero_si256();
            let mut c20 = _mm256_setzero_si256();
            let mut c21 = _mm256_setzero_si256();
            let mut c30 = _mm256_setzero_si256();
            let mut c31 = _mm256_setzero_si256();
            let mut c40 = _mm256_setzero_si256();
            let mut c41 = _mm256_setzero_si256();
            let mut c50 = _mm256_setzero_si256();
            let mut c51 = _mm256_setzero_si256();

            let mut bp = block;
            for k4 in 0..32usize {
                let b0 = _mm256_loadu_si256(bp as *const __m256i);
                let b1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
                bp = bp.add(64);

                let av0 = _mm256_set1_epi32(*a0.add(k4));
                c00 = _mm256_dpbusd_avx_epi32(c00, av0, b0);
                c01 = _mm256_dpbusd_avx_epi32(c01, av0, b1);

                let av1 = _mm256_set1_epi32(*a1.add(k4));
                c10 = _mm256_dpbusd_avx_epi32(c10, av1, b0);
                c11 = _mm256_dpbusd_avx_epi32(c11, av1, b1);

                let av2 = _mm256_set1_epi32(*a2.add(k4));
                c20 = _mm256_dpbusd_avx_epi32(c20, av2, b0);
                c21 = _mm256_dpbusd_avx_epi32(c21, av2, b1);

                let av3 = _mm256_set1_epi32(*a3.add(k4));
                c30 = _mm256_dpbusd_avx_epi32(c30, av3, b0);
                c31 = _mm256_dpbusd_avx_epi32(c31, av3, b1);

                let av4 = _mm256_set1_epi32(*a4.add(k4));
                c40 = _mm256_dpbusd_avx_epi32(c40, av4, b0);
                c41 = _mm256_dpbusd_avx_epi32(c41, av4, b1);

                let av5 = _mm256_set1_epi32(*a5.add(k4));
                c50 = _mm256_dpbusd_avx_epi32(c50, av5, b0);
                c51 = _mm256_dpbusd_avx_epi32(c51, av5, b1);
            }

            let wb0 = _mm256_loadu_si256(block.add(2048) as *const __m256i);
            let wb1 = _mm256_loadu_si256(block.add(2080) as *const __m256i);
            let ws0 = _mm256_loadu_ps(block.add(2112) as *const f32);
            let ws1 = _mm256_loadu_ps(block.add(2144) as *const f32);

            macro_rules! accum_row {
                ($r:expr, $ci0:expr, $ci1:expr) => {
                    let sa = _mm256_set1_ps(*s_ptr.add($r * k_groups + g));
                    let sc0 = _mm256_mul_ps(sa, ws0);
                    let sc1 = _mm256_mul_ps(sa, ws1);
                    let f0 = _mm256_cvtepi32_ps(_mm256_sub_epi32($ci0, wb0));
                    let f1 = _mm256_cvtepi32_ps(_mm256_sub_epi32($ci1, wb1));
                    let dst = facc[$r].as_mut_ptr();
                    let cur0 = _mm256_loadu_ps(dst);
                    let cur1 = _mm256_loadu_ps(dst.add(8));
                    _mm256_storeu_ps(dst, _mm256_fmadd_ps(f0, sc0, cur0));
                    _mm256_storeu_ps(dst.add(8), _mm256_fmadd_ps(f1, sc1, cur1));
                };
            }
            accum_row!(0, c00, c01);
            accum_row!(1, c10, c11);
            accum_row!(2, c20, c21);
            accum_row!(3, c30, c31);
            accum_row!(4, c40, c41);
            accum_row!(5, c50, c51);
        }

        store_tile_f32(&facc, 6, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

/// Ultra-fast single-group (`k_groups == 1`, `K = 128`) 6x16 AVX-VNNI microkernel:
/// keeps everything in registers with ZERO stack `facc` load/store!
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn microkernel_q8_6x16_vnni_1g(
    a_ptr: *const u8,
    s_ptr: *const f32,
    block: *const u8,
    o_ptr: *mut f32,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let a0 = a_ptr as *const i32;
        let a1 = a_ptr.add(GROUP_SIZE) as *const i32;
        let a2 = a_ptr.add(GROUP_SIZE * 2) as *const i32;
        let a3 = a_ptr.add(GROUP_SIZE * 3) as *const i32;
        let a4 = a_ptr.add(GROUP_SIZE * 4) as *const i32;
        let a5 = a_ptr.add(GROUP_SIZE * 5) as *const i32;

        let mut c00 = _mm256_setzero_si256();
        let mut c01 = _mm256_setzero_si256();
        let mut c10 = _mm256_setzero_si256();
        let mut c11 = _mm256_setzero_si256();
        let mut c20 = _mm256_setzero_si256();
        let mut c21 = _mm256_setzero_si256();
        let mut c30 = _mm256_setzero_si256();
        let mut c31 = _mm256_setzero_si256();
        let mut c40 = _mm256_setzero_si256();
        let mut c41 = _mm256_setzero_si256();
        let mut c50 = _mm256_setzero_si256();
        let mut c51 = _mm256_setzero_si256();

        let mut bp = block;
        for k4 in 0..32usize {
            let b0 = _mm256_loadu_si256(bp as *const __m256i);
            let b1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
            bp = bp.add(64);

            let av0 = _mm256_set1_epi32(*a0.add(k4));
            c00 = _mm256_dpbusd_avx_epi32(c00, av0, b0);
            c01 = _mm256_dpbusd_avx_epi32(c01, av0, b1);

            let av1 = _mm256_set1_epi32(*a1.add(k4));
            c10 = _mm256_dpbusd_avx_epi32(c10, av1, b0);
            c11 = _mm256_dpbusd_avx_epi32(c11, av1, b1);

            let av2 = _mm256_set1_epi32(*a2.add(k4));
            c20 = _mm256_dpbusd_avx_epi32(c20, av2, b0);
            c21 = _mm256_dpbusd_avx_epi32(c21, av2, b1);

            let av3 = _mm256_set1_epi32(*a3.add(k4));
            c30 = _mm256_dpbusd_avx_epi32(c30, av3, b0);
            c31 = _mm256_dpbusd_avx_epi32(c31, av3, b1);

            let av4 = _mm256_set1_epi32(*a4.add(k4));
            c40 = _mm256_dpbusd_avx_epi32(c40, av4, b0);
            c41 = _mm256_dpbusd_avx_epi32(c41, av4, b1);

            let av5 = _mm256_set1_epi32(*a5.add(k4));
            c50 = _mm256_dpbusd_avx_epi32(c50, av5, b0);
            c51 = _mm256_dpbusd_avx_epi32(c51, av5, b1);
        }

        let wb0 = _mm256_loadu_si256(block.add(2048) as *const __m256i);
        let wb1 = _mm256_loadu_si256(block.add(2080) as *const __m256i);
        let ws0 = _mm256_loadu_ps(block.add(2112) as *const f32);
        let ws1 = _mm256_loadu_ps(block.add(2144) as *const f32);

        macro_rules! store_row {
            ($r:expr, $ci0:expr, $ci1:expr) => {
                let sa = _mm256_set1_ps(*s_ptr.add($r));
                let v0 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32($ci0, wb0)),
                    _mm256_mul_ps(sa, ws0),
                );
                let v1 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32($ci1, wb1)),
                    _mm256_mul_ps(sa, ws1),
                );
                let dst = o_ptr.add($r * n_stride);
                if n_valid == 16 {
                    _mm256_storeu_ps(dst, v0);
                    _mm256_storeu_ps(dst.add(8), v1);
                } else {
                    let mut tmp = [0.0f32; 16];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v0);
                    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), v1);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, n_valid);
                }
            };
        }
        store_row!(0, c00, c01);
        store_row!(1, c10, c11);
        store_row!(2, c20, c21);
        store_row!(3, c30, c31);
        store_row!(4, c40, c41);
        store_row!(5, c50, c51);
    }
}

/// Remainder M (1..5 rows) x 16 AVX-VNNI Q8K128 microkernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn microkernel_q8_remx16_vnni(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    rem: usize,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        for r in 0..rem {
            let mut f0 = _mm256_setzero_ps();
            let mut f1 = _mm256_setzero_ps();
            let ar_base = a_ptr.add(r * k_pad);
            for g in 0..k_groups {
                let block = w_tile.add(g * Q8K128_BLOCK_BYTES);
                let ar = ar_base.add(g * GROUP_SIZE) as *const i32;
                let mut c0 = _mm256_setzero_si256();
                let mut c1 = _mm256_setzero_si256();
                let mut bp = block;
                for k4 in 0..32usize {
                    let b0 = _mm256_loadu_si256(bp as *const __m256i);
                    let b1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
                    bp = bp.add(64);
                    let av = _mm256_set1_epi32(*ar.add(k4));
                    c0 = _mm256_dpbusd_avx_epi32(c0, av, b0);
                    c1 = _mm256_dpbusd_avx_epi32(c1, av, b1);
                }
                let wb0 = _mm256_loadu_si256(block.add(2048) as *const __m256i);
                let wb1 = _mm256_loadu_si256(block.add(2080) as *const __m256i);
                let ws0 = _mm256_loadu_ps(block.add(2112) as *const f32);
                let ws1 = _mm256_loadu_ps(block.add(2144) as *const f32);
                let sa = _mm256_set1_ps(*s_ptr.add(r * k_groups + g));
                f0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(c0, wb0)),
                    _mm256_mul_ps(sa, ws0),
                    f0,
                );
                f1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(c1, wb1)),
                    _mm256_mul_ps(sa, ws1),
                    f1,
                );
            }
            _mm256_storeu_ps(facc[r].as_mut_ptr(), f0);
            _mm256_storeu_ps(facc[r].as_mut_ptr().add(8), f1);
        }
        store_tile_f32(&facc, rem, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

/// AVX2 (non-VNNI) fallback 6x16 Q8K128 microkernel using `vpmaddubsw` + `vpmaddwd`.
/// Note: since `a_u8` is in `[1, 255]` and `b_i8` in `[-127, 127]`, a pair product sum can reach
/// `2 * 255 * 127 = 64770` which saturates i16! So for non-VNNI AVX2 we subtract 128 from `a`
/// (giving signed `a_i8` in `[-127, 127]`) and use `sign(a, b)` -> `maddubs(|a|, sign(b, a))`,
/// where max pair sum is `2 * 127 * 127 = 32258 <= 32767` (guaranteed no i16 saturation!).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_q8_6x16_avx2(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        let ones = _mm256_set1_epi16(1);
        let v128 = _mm256_set1_epi8(-128i8);
        for g in 0..k_groups {
            let block = w_tile.add(g * Q8K128_BLOCK_BYTES);
            let k_off = g * GROUP_SIZE;
            let ws0 = _mm256_loadu_ps(block.add(2112) as *const f32);
            let ws1 = _mm256_loadu_ps(block.add(2144) as *const f32);
            for r in 0..6 {
                let ar = a_ptr.add(r * k_pad + k_off) as *const i32;
                let mut c0 = _mm256_setzero_si256();
                let mut c1 = _mm256_setzero_si256();
                let mut bp = block;
                for k4 in 0..32usize {
                    let b0 = _mm256_loadu_si256(bp as *const __m256i);
                    let b1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
                    bp = bp.add(64);
                    let av_s = _mm256_sub_epi8(_mm256_set1_epi32(*ar.add(k4)), v128);
                    let av_abs = _mm256_abs_epi8(av_s);
                    let b0_s = _mm256_sign_epi8(b0, av_s);
                    let b1_s = _mm256_sign_epi8(b1, av_s);
                    let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(av_abs, b0_s), ones);
                    let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(av_abs, b1_s), ones);
                    c0 = _mm256_add_epi32(c0, p0);
                    c1 = _mm256_add_epi32(c1, p1);
                }
                let sa = _mm256_set1_ps(*s_ptr.add(r * k_groups + g));
                let dst = facc[r].as_mut_ptr();
                let cur0 = _mm256_loadu_ps(dst);
                let cur1 = _mm256_loadu_ps(dst.add(8));
                _mm256_storeu_ps(
                    dst,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_mul_ps(sa, ws0), cur0),
                );
                _mm256_storeu_ps(
                    dst.add(8),
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_mul_ps(sa, ws1), cur1),
                );
            }
        }
        store_tile_f32(&facc, 6, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

unsafe fn microkernel_q8_scalar(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    rows: usize,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        for r in 0..rows {
            for g in 0..k_groups {
                let block = w_tile.add(g * Q8K128_BLOCK_BYTES);
                let ar = a_ptr.add(r * k_pad + g * GROUP_SIZE);
                let sa = *s_ptr.add(r * k_groups + g);
                let wb = block.add(2048) as *const i32;
                let ws = block.add(2112) as *const f32;
                for c in 0..n_valid {
                    let half_off = if c < 8 { c * 4 } else { 32 + (c - 8) * 4 };
                    let mut sum = 0i32;
                    for k4 in 0..32 {
                        let bp = block.add(k4 * 64 + half_off);
                        for j in 0..4 {
                            let av = *ar.add(k4 * 4 + j) as i32;
                            let bv = (*bp.add(j) as i8) as i32;
                            sum += av * bv;
                        }
                    }
                    facc[r][c] += (sum - *wb.add(c)) as f32 * (sa * *ws.add(c));
                }
            }
        }
        store_tile_f32(&facc, rows, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

/// Parallel tiled Q4K128 GEMM (4-bit weights + 8-bit dynamic activations via AVX-VNNI).
fn matmul_q4k128_parallel(
    act: &ActQuantBuf,
    w_data: &[u8],
    n: usize,
    n_tiles: usize,
    k_groups: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let m = act.m;
    let k_pad = act.k_pad;
    let tile_stride = k_groups * Q4K128_BLOCK_BYTES;

    #[cfg(target_arch = "x86_64")]
    let use_vnni = std::arch::is_x86_feature_detected!("avxvnni");
    #[cfg(not(target_arch = "x86_64"))]
    let use_vnni = false;

    let mc = if m >= 1024 {
        258
    } else if m >= 256 {
        126
    } else {
        m.max(1)
    };
    let m_blocks = (m + mc - 1) / mc;
    let total_tasks = m_blocks * n_tiles;

    let out_ptr = out.as_mut_ptr() as usize;
    let a_ptr = act.q_u8.as_ptr() as usize;
    let s_ptr = act.scales.as_ptr() as usize;
    let w_ptr = w_data.as_ptr() as usize;
    let b_ptr = bias.map(|b| b.as_ptr() as usize).unwrap_or(0);

    let run_task = |task_id: usize| {
        let mb = task_id / n_tiles;
        let nt = task_id % n_tiles;
        let m0 = mb * mc;
        let m1 = (m0 + mc).min(m);
        let n0 = nt * NR_TILE;
        let n_valid = (n - n0).min(NR_TILE);
        let w_tile = unsafe { (w_ptr as *const u8).add(nt * tile_stride) };
        let a_base = a_ptr as *const u8;
        let s_base = s_ptr as *const f32;
        let o_base = out_ptr as *mut f32;
        let bias_ptr = if b_ptr != 0 {
            unsafe { Some((b_ptr as *const f32).add(n0)) }
        } else {
            None
        };

        let mut mr_cur = m0;
        while mr_cur + 6 <= m1 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_vnni {
                    microkernel_q4_6x16_vnni(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                } else {
                    microkernel_q4_scalar(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        6,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe {
                microkernel_q4_scalar(
                    a_base.add(mr_cur * k_pad),
                    s_base.add(mr_cur * k_groups),
                    w_tile,
                    o_base.add(mr_cur * n + n0),
                    bias_ptr,
                    6,
                    k_pad,
                    k_groups,
                    n,
                    n_valid,
                );
            }
            mr_cur += 6;
        }
        let rem = m1 - mr_cur;
        if rem > 0 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_vnni {
                    microkernel_q4_remx16_vnni(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        rem,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                } else {
                    microkernel_q4_scalar(
                        a_base.add(mr_cur * k_pad),
                        s_base.add(mr_cur * k_groups),
                        w_tile,
                        o_base.add(mr_cur * n + n0),
                        bias_ptr,
                        rem,
                        k_pad,
                        k_groups,
                        n,
                        n_valid,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe {
                microkernel_q4_scalar(
                    a_base.add(mr_cur * k_pad),
                    s_base.add(mr_cur * k_groups),
                    w_tile,
                    o_base.add(mr_cur * n + n0),
                    bias_ptr,
                    rem,
                    k_pad,
                    k_groups,
                    n,
                    n_valid,
                );
            }
        }
    };

    if total_tasks >= 8 && m * n * k_groups >= 65536 {
        (0..total_tasks).into_par_iter().for_each(run_task);
    } else {
        for t in 0..total_tasks {
            run_task(t);
        }
    }
}

/// Core 6x16 AVX-VNNI Q4K128 microkernel: unpacks 4-bit nibbles on-the-fly in registers!
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn microkernel_q4_6x16_vnni(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        let mask_0f = _mm256_set1_epi8(0x0Fi8);
        let v8 = _mm256_set1_epi8(8i8);

        let a0_base = a_ptr;
        let a1_base = a_ptr.add(k_pad);
        let a2_base = a_ptr.add(k_pad * 2);
        let a3_base = a_ptr.add(k_pad * 3);
        let a4_base = a_ptr.add(k_pad * 4);
        let a5_base = a_ptr.add(k_pad * 5);

        for g in 0..k_groups {
            let block = w_tile.add(g * Q4K128_BLOCK_BYTES);
            let k_off = g * GROUP_SIZE;

            let a0 = a0_base.add(k_off) as *const i32;
            let a1 = a1_base.add(k_off) as *const i32;
            let a2 = a2_base.add(k_off) as *const i32;
            let a3 = a3_base.add(k_off) as *const i32;
            let a4 = a4_base.add(k_off) as *const i32;
            let a5 = a5_base.add(k_off) as *const i32;

            let mut c00 = _mm256_setzero_si256();
            let mut c01 = _mm256_setzero_si256();
            let mut c10 = _mm256_setzero_si256();
            let mut c11 = _mm256_setzero_si256();
            let mut c20 = _mm256_setzero_si256();
            let mut c21 = _mm256_setzero_si256();
            let mut c30 = _mm256_setzero_si256();
            let mut c31 = _mm256_setzero_si256();
            let mut c40 = _mm256_setzero_si256();
            let mut c41 = _mm256_setzero_si256();
            let mut c50 = _mm256_setzero_si256();
            let mut c51 = _mm256_setzero_si256();

            let mut bp = block;
            for p in 0..16usize {
                let pk0 = _mm256_loadu_si256(bp as *const __m256i);
                let pk1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
                bp = bp.add(64);

                // Low nibble: k4 = 2*p
                let b0_lo = _mm256_sub_epi8(_mm256_and_si256(pk0, mask_0f), v8);
                let b1_lo = _mm256_sub_epi8(_mm256_and_si256(pk1, mask_0f), v8);
                let k_lo = 2 * p;

                let av0 = _mm256_set1_epi32(*a0.add(k_lo));
                c00 = _mm256_dpbusd_avx_epi32(c00, av0, b0_lo);
                c01 = _mm256_dpbusd_avx_epi32(c01, av0, b1_lo);

                let av1 = _mm256_set1_epi32(*a1.add(k_lo));
                c10 = _mm256_dpbusd_avx_epi32(c10, av1, b0_lo);
                c11 = _mm256_dpbusd_avx_epi32(c11, av1, b1_lo);

                let av2 = _mm256_set1_epi32(*a2.add(k_lo));
                c20 = _mm256_dpbusd_avx_epi32(c20, av2, b0_lo);
                c21 = _mm256_dpbusd_avx_epi32(c21, av2, b1_lo);

                let av3 = _mm256_set1_epi32(*a3.add(k_lo));
                c30 = _mm256_dpbusd_avx_epi32(c30, av3, b0_lo);
                c31 = _mm256_dpbusd_avx_epi32(c31, av3, b1_lo);

                let av4 = _mm256_set1_epi32(*a4.add(k_lo));
                c40 = _mm256_dpbusd_avx_epi32(c40, av4, b0_lo);
                c41 = _mm256_dpbusd_avx_epi32(c41, av4, b1_lo);

                let av5 = _mm256_set1_epi32(*a5.add(k_lo));
                c50 = _mm256_dpbusd_avx_epi32(c50, av5, b0_lo);
                c51 = _mm256_dpbusd_avx_epi32(c51, av5, b1_lo);

                // High nibble: k4 = 2*p + 1
                let b0_hi =
                    _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(pk0, 4), mask_0f), v8);
                let b1_hi =
                    _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(pk1, 4), mask_0f), v8);
                let k_hi = 2 * p + 1;

                let av0 = _mm256_set1_epi32(*a0.add(k_hi));
                c00 = _mm256_dpbusd_avx_epi32(c00, av0, b0_hi);
                c01 = _mm256_dpbusd_avx_epi32(c01, av0, b1_hi);

                let av1 = _mm256_set1_epi32(*a1.add(k_hi));
                c10 = _mm256_dpbusd_avx_epi32(c10, av1, b0_hi);
                c11 = _mm256_dpbusd_avx_epi32(c11, av1, b1_hi);

                let av2 = _mm256_set1_epi32(*a2.add(k_hi));
                c20 = _mm256_dpbusd_avx_epi32(c20, av2, b0_hi);
                c21 = _mm256_dpbusd_avx_epi32(c21, av2, b1_hi);

                let av3 = _mm256_set1_epi32(*a3.add(k_hi));
                c30 = _mm256_dpbusd_avx_epi32(c30, av3, b0_hi);
                c31 = _mm256_dpbusd_avx_epi32(c31, av3, b1_hi);

                let av4 = _mm256_set1_epi32(*a4.add(k_hi));
                c40 = _mm256_dpbusd_avx_epi32(c40, av4, b0_hi);
                c41 = _mm256_dpbusd_avx_epi32(c41, av4, b1_hi);

                let av5 = _mm256_set1_epi32(*a5.add(k_hi));
                c50 = _mm256_dpbusd_avx_epi32(c50, av5, b0_hi);
                c51 = _mm256_dpbusd_avx_epi32(c51, av5, b1_hi);
            }

            let wb0 = _mm256_loadu_si256(block.add(1024) as *const __m256i);
            let wb1 = _mm256_loadu_si256(block.add(1056) as *const __m256i);
            let ws0 = _mm256_loadu_ps(block.add(1088) as *const f32);
            let ws1 = _mm256_loadu_ps(block.add(1120) as *const f32);

            macro_rules! accum_row {
                ($r:expr, $ci0:expr, $ci1:expr) => {
                    let sa = _mm256_set1_ps(*s_ptr.add($r * k_groups + g));
                    let sc0 = _mm256_mul_ps(sa, ws0);
                    let sc1 = _mm256_mul_ps(sa, ws1);
                    let f0 = _mm256_cvtepi32_ps(_mm256_sub_epi32($ci0, wb0));
                    let f1 = _mm256_cvtepi32_ps(_mm256_sub_epi32($ci1, wb1));
                    let dst = facc[$r].as_mut_ptr();
                    let cur0 = _mm256_loadu_ps(dst);
                    let cur1 = _mm256_loadu_ps(dst.add(8));
                    _mm256_storeu_ps(dst, _mm256_fmadd_ps(f0, sc0, cur0));
                    _mm256_storeu_ps(dst.add(8), _mm256_fmadd_ps(f1, sc1, cur1));
                };
            }
            accum_row!(0, c00, c01);
            accum_row!(1, c10, c11);
            accum_row!(2, c20, c21);
            accum_row!(3, c30, c31);
            accum_row!(4, c40, c41);
            accum_row!(5, c50, c51);
        }

        store_tile_f32(&facc, 6, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn microkernel_q4_remx16_vnni(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    rem: usize,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        let mask_0f = _mm256_set1_epi8(0x0Fi8);
        let v8 = _mm256_set1_epi8(8i8);
        for r in 0..rem {
            let mut f0 = _mm256_setzero_ps();
            let mut f1 = _mm256_setzero_ps();
            let ar_base = a_ptr.add(r * k_pad);
            for g in 0..k_groups {
                let block = w_tile.add(g * Q4K128_BLOCK_BYTES);
                let ar = ar_base.add(g * GROUP_SIZE) as *const i32;
                let mut c0 = _mm256_setzero_si256();
                let mut c1 = _mm256_setzero_si256();
                let mut bp = block;
                for p in 0..16usize {
                    let pk0 = _mm256_loadu_si256(bp as *const __m256i);
                    let pk1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);
                    bp = bp.add(64);
                    let b0_lo = _mm256_sub_epi8(_mm256_and_si256(pk0, mask_0f), v8);
                    let b1_lo = _mm256_sub_epi8(_mm256_and_si256(pk1, mask_0f), v8);
                    let av_lo = _mm256_set1_epi32(*ar.add(2 * p));
                    c0 = _mm256_dpbusd_avx_epi32(c0, av_lo, b0_lo);
                    c1 = _mm256_dpbusd_avx_epi32(c1, av_lo, b1_lo);

                    let b0_hi =
                        _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(pk0, 4), mask_0f), v8);
                    let b1_hi =
                        _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(pk1, 4), mask_0f), v8);
                    let av_hi = _mm256_set1_epi32(*ar.add(2 * p + 1));
                    c0 = _mm256_dpbusd_avx_epi32(c0, av_hi, b0_hi);
                    c1 = _mm256_dpbusd_avx_epi32(c1, av_hi, b1_hi);
                }
                let wb0 = _mm256_loadu_si256(block.add(1024) as *const __m256i);
                let wb1 = _mm256_loadu_si256(block.add(1056) as *const __m256i);
                let ws0 = _mm256_loadu_ps(block.add(1088) as *const f32);
                let ws1 = _mm256_loadu_ps(block.add(1120) as *const f32);
                let sa = _mm256_set1_ps(*s_ptr.add(r * k_groups + g));
                f0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(c0, wb0)),
                    _mm256_mul_ps(sa, ws0),
                    f0,
                );
                f1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(c1, wb1)),
                    _mm256_mul_ps(sa, ws1),
                    f1,
                );
            }
            _mm256_storeu_ps(facc[r].as_mut_ptr(), f0);
            _mm256_storeu_ps(facc[r].as_mut_ptr().add(8), f1);
        }
        store_tile_f32(&facc, rem, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

unsafe fn microkernel_q4_scalar(
    a_ptr: *const u8,
    s_ptr: *const f32,
    w_tile: *const u8,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    rows: usize,
    k_pad: usize,
    k_groups: usize,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        let mut facc = [[0.0f32; 16]; 6];
        for r in 0..rows {
            for g in 0..k_groups {
                let block = w_tile.add(g * Q4K128_BLOCK_BYTES);
                let ar = a_ptr.add(r * k_pad + g * GROUP_SIZE);
                let sa = *s_ptr.add(r * k_groups + g);
                let wb = block.add(1024) as *const i32;
                let ws = block.add(1088) as *const f32;
                for c in 0..n_valid {
                    let half_off = if c < 8 { c * 4 } else { 32 + (c - 8) * 4 };
                    let mut sum = 0i32;
                    for p in 0..16 {
                        let bp = block.add(p * 64 + half_off);
                        for j in 0..4 {
                            let packed = *bp.add(j);
                            let q_lo = (packed & 0x0F) as i32 - 8;
                            let q_hi = ((packed >> 4) & 0x0F) as i32 - 8;
                            let a_lo = *ar.add((2 * p) * 4 + j) as i32;
                            let a_hi = *ar.add((2 * p + 1) * 4 + j) as i32;
                            sum += a_lo * q_lo + a_hi * q_hi;
                        }
                    }
                    facc[r][c] += (sum - *wb.add(c)) as f32 * (sa * *ws.add(c));
                }
            }
        }
        store_tile_f32(&facc, rows, o_ptr, bias_ptr, n_stride, n_valid);
    }
}

#[inline(always)]
unsafe fn store_tile_f32(
    facc: &[[f32; 16]; 6],
    rows: usize,
    o_ptr: *mut f32,
    bias_ptr: Option<*const f32>,
    n_stride: usize,
    n_valid: usize,
) {
    unsafe {
        if n_valid == 16 {
            for r in 0..rows {
                let dst = o_ptr.add(r * n_stride);
                if let Some(bp) = bias_ptr {
                    for c in 0..16 {
                        *dst.add(c) = facc[r][c] + *bp.add(c);
                    }
                } else {
                    std::ptr::copy_nonoverlapping(facc[r].as_ptr(), dst, 16);
                }
            }
        } else {
            for r in 0..rows {
                let dst = o_ptr.add(r * n_stride);
                if let Some(bp) = bias_ptr {
                    for c in 0..n_valid {
                        *dst.add(c) = facc[r][c] + *bp.add(c);
                    }
                } else {
                    std::ptr::copy_nonoverlapping(facc[r].as_ptr(), dst, n_valid);
                }
            }
        }
    }
}

/// Fast F32 matmul `out[M, N] = a[M, K] @ b[N, K]^T (+ bias)` for small/sensitive F32 layers.
pub fn matmul_f32(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(out.len(), m * n);

    let compute_row = |r: usize, orow: &mut [f32]| {
        let arow = &a[r * k..(r + 1) * k];
        for c in 0..n {
            let brow = &b[c * k..(c + 1) * k];
            let mut acc = dot_f32(arow, brow);
            if let Some(bi) = bias {
                acc += bi[c];
            }
            orow[c] = acc;
        }
    };

    if m >= 16 && m * n * k >= 32768 {
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(r, orow)| compute_row(r, orow));
    } else {
        for r in 0..m {
            compute_row(r, &mut out[r * n..(r + 1) * n]);
        }
    }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 8)),
                _mm256_loadu_ps(bp.add(i + 8)),
                s1,
            );
            s2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 16)),
                _mm256_loadu_ps(bp.add(i + 16)),
                s2,
            );
            s3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 24)),
                _mm256_loadu_ps(bp.add(i + 24)),
                s3,
            );
            i += 32;
        }
        s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(s0, 1);
        let lo = _mm256_castps256_ps128(s0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        sum
    }
}

/// Fused axpy: `dst[i] += alpha * src[i]` using AVX2 FMA.
#[inline]
pub fn axpy_f32(dst: &mut [f32], src: &[f32], alpha: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                axpy_f32_avx2(dst, src, alpha);
            }
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(dst: &mut [f32], src: &[f32], alpha: f32) {
    unsafe {
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let va = _mm256_set1_ps(alpha);
        let mut i = 0;
        while i + 32 <= n {
            let d0 = _mm256_loadu_ps(dp.add(i));
            let d1 = _mm256_loadu_ps(dp.add(i + 8));
            let d2 = _mm256_loadu_ps(dp.add(i + 16));
            let d3 = _mm256_loadu_ps(dp.add(i + 24));
            _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(sp.add(i)), d0));
            _mm256_storeu_ps(
                dp.add(i + 8),
                _mm256_fmadd_ps(va, _mm256_loadu_ps(sp.add(i + 8)), d1),
            );
            _mm256_storeu_ps(
                dp.add(i + 16),
                _mm256_fmadd_ps(va, _mm256_loadu_ps(sp.add(i + 16)), d2),
            );
            _mm256_storeu_ps(
                dp.add(i + 24),
                _mm256_fmadd_ps(va, _mm256_loadu_ps(sp.add(i + 24)), d3),
            );
            i += 32;
        }
        while i + 8 <= n {
            let d0 = _mm256_loadu_ps(dp.add(i));
            _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(sp.add(i)), d0));
            i += 8;
        }
        while i < n {
            *dp.add(i) += alpha * *sp.add(i);
            i += 1;
        }
    }
}

/// Fast SiLU activation: `x / (1 + exp(-x))`
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 8-wide AVX2 fast `exp(x)` with range reduction + degree-5 minimax polynomial (max rel err < 2e-7).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp256_ps(x: __m256) -> __m256 {
    let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949);
    let x = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);

    let log2e = _mm256_set1_ps(1.4426950408889634);
    let ln2_hi = _mm256_set1_ps(0.693359375);
    let ln2_lo = _mm256_set1_ps(-2.12194440e-4);

    // k = round(x * log2(e))
    let fx = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    // r = x - k * ln2
    let r = _mm256_fnmadd_ps(fx, ln2_lo, _mm256_fnmadd_ps(fx, ln2_hi, x));

    // Degree-5 Taylor/minimax for exp(r) on [-ln2/2, ln2/2]
    let p5 = _mm256_set1_ps(1.0 / 120.0);
    let p4 = _mm256_set1_ps(1.0 / 24.0);
    let p3 = _mm256_set1_ps(1.0 / 6.0);
    let p2 = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);

    let mut y = _mm256_fmadd_ps(p5, r, p4);
    y = _mm256_fmadd_ps(y, r, p3);
    y = _mm256_fmadd_ps(y, r, p2);
    y = _mm256_fmadd_ps(y, r, one);
    y = _mm256_fmadd_ps(y, r, one);

    // 2^k via IEEE-754 exponent
    let k_i32 = _mm256_cvtps_epi32(fx);
    let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(k_i32, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(y, pow2)
}

/// Vectorized SwiGLU: `dst[i] = silu(gate[i]) * up[i]`
#[inline]
pub fn swiglu_row(gate: &[f32], up: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                swiglu_row_avx2(gate, up, dst);
            }
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] = silu(gate[i]) * up[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_row_avx2(gate: &[f32], up: &[f32], dst: &mut [f32]) {
    unsafe {
        let n = dst.len();
        let gp = gate.as_ptr();
        let up_p = up.as_ptr();
        let dp = dst.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let neg_zero = _mm256_set1_ps(-0.0);
        let mut i = 0;
        while i + 8 <= n {
            let g = _mm256_loadu_ps(gp.add(i));
            let u = _mm256_loadu_ps(up_p.add(i));
            let neg_g = _mm256_xor_ps(g, neg_zero);
            let e = exp256_ps(neg_g);
            let sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
            _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(sig, u));
            i += 8;
        }
        while i < n {
            *dp.add(i) = silu(*gp.add(i)) * *up_p.add(i);
            i += 1;
        }
    }
}

/// Vectorized in-place softmax over `scores[..n]` using AVX2 fast exp.
#[inline]
pub fn softmax_inplace(scores: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                softmax_inplace_avx2(scores);
            }
            return;
        }
    }
    let mut max_s = f32::NEG_INFINITY;
    for &v in scores.iter() {
        if v > max_s {
            max_s = v;
        }
    }
    let mut sum = 0.0f32;
    for v in scores.iter_mut() {
        let e = (*v - max_s).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in scores.iter_mut() {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_inplace_avx2(scores: &mut [f32]) {
    unsafe {
        let n = scores.len();
        let p = scores.as_mut_ptr();
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= n {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(p.add(i)));
            i += 8;
        }
        let hi = _mm256_extractf128_ps(vmax, 1);
        let lo = _mm256_castps256_ps128(vmax);
        let m128 = _mm_max_ps(lo, hi);
        let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        let mut max_s = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1)));
        while i < n {
            let v = *p.add(i);
            if v > max_s {
                max_s = v;
            }
            i += 1;
        }

        let vmax_b = _mm256_set1_ps(max_s);
        let mut vsum = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), vmax_b));
            _mm256_storeu_ps(p.add(i), e);
            vsum = _mm256_add_ps(vsum, e);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(vsum, 1);
        let lo = _mm256_castps256_ps128(vsum);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
        while i < n {
            let e = (*p.add(i) - max_s).exp();
            *p.add(i) = e;
            sum += e;
            i += 1;
        }

        let inv = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), vinv));
            i += 8;
        }
        while i < n {
            *p.add(i) *= inv;
            i += 1;
        }
    }
}

/// 4-query dot product against a single key `k`: computes `(q0·k, q1·k, q2·k, q3·k)`
/// reading `k` only ONCE from cache!
#[inline]
pub fn dot4_f32(q0: &[f32], q1: &[f32], q2: &[f32], q3: &[f32], k: &[f32]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot4_f32_avx2(q0, q1, q2, q3, k) };
        }
    }
    [
        dot_f32(q0, k),
        dot_f32(q1, k),
        dot_f32(q2, k),
        dot_f32(q3, k),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_f32_avx2(q0: &[f32], q1: &[f32], q2: &[f32], q3: &[f32], k: &[f32]) -> [f32; 4] {
    unsafe {
        let n = k.len();
        let kp = k.as_ptr();
        let p0 = q0.as_ptr();
        let p1 = q1.as_ptr();
        let p2 = q2.as_ptr();
        let p3 = q3.as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let kv0 = _mm256_loadu_ps(kp.add(i));
            let kv1 = _mm256_loadu_ps(kp.add(i + 8));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i)), kv0, a0);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i + 8)), kv1, b0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i)), kv0, a1);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i + 8)), kv1, b1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(i)), kv0, a2);
            b2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(i + 8)), kv1, b2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(i)), kv0, a3);
            b3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(i + 8)), kv1, b3);
            i += 16;
        }
        a0 = _mm256_add_ps(a0, b0);
        a1 = _mm256_add_ps(a1, b1);
        a2 = _mm256_add_ps(a2, b2);
        a3 = _mm256_add_ps(a3, b3);
        let hsum = |v: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let s = _mm_add_ps(lo, hi);
            let sh = _mm_movehdup_ps(s);
            let ss = _mm_add_ps(s, sh);
            _mm_cvtss_f32(_mm_add_ss(ss, _mm_movehl_ps(sh, ss)))
        };
        [hsum(a0), hsum(a1), hsum(a2), hsum(a3)]
    }
}

/// 4-query value accumulation: `acc[r] += w[r] * v` reading `v` only ONCE from cache!
#[inline]
pub fn axpy4_f32(
    d0: &mut [f32],
    d1: &mut [f32],
    d2: &mut [f32],
    d3: &mut [f32],
    v: &[f32],
    w: [f32; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                axpy4_f32_avx2(d0, d1, d2, d3, v, w);
            }
            return;
        }
    }
    axpy_f32(d0, v, w[0]);
    axpy_f32(d1, v, w[1]);
    axpy_f32(d2, v, w[2]);
    axpy_f32(d3, v, w[3]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy4_f32_avx2(
    d0: &mut [f32],
    d1: &mut [f32],
    d2: &mut [f32],
    d3: &mut [f32],
    v: &[f32],
    w: [f32; 4],
) {
    unsafe {
        let n = v.len();
        let vp = v.as_ptr();
        let p0 = d0.as_mut_ptr();
        let p1 = d1.as_mut_ptr();
        let p2 = d2.as_mut_ptr();
        let p3 = d3.as_mut_ptr();
        let w0 = _mm256_set1_ps(w[0]);
        let w1 = _mm256_set1_ps(w[1]);
        let w2 = _mm256_set1_ps(w[2]);
        let w3 = _mm256_set1_ps(w[3]);
        let mut i = 0;
        while i + 8 <= n {
            let vv = _mm256_loadu_ps(vp.add(i));
            _mm256_storeu_ps(p0.add(i), _mm256_fmadd_ps(w0, vv, _mm256_loadu_ps(p0.add(i))));
            _mm256_storeu_ps(p1.add(i), _mm256_fmadd_ps(w1, vv, _mm256_loadu_ps(p1.add(i))));
            _mm256_storeu_ps(p2.add(i), _mm256_fmadd_ps(w2, vv, _mm256_loadu_ps(p2.add(i))));
            _mm256_storeu_ps(p3.add(i), _mm256_fmadd_ps(w3, vv, _mm256_loadu_ps(p3.add(i))));
            i += 8;
        }
    }
}

/// Fast GELU (tanh approximation), matching PyTorch `nn.GELU(approximate="tanh")`
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`
#[inline]
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let ss = dot_f32(x, x) / (n as f32);
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv * weight[i];
    }
}

/// Zero-centered RMSNorm: `x * rsqrt(mean(x^2) + eps) * (weight + 1.0)`
#[inline]
pub fn zero_center_rms_norm(src: &[f32], dst: &mut [f32], weight: &[f32], eps: f32) {
    let n = src.len();
    let ss = dot_f32(src, src) / (n as f32);
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        dst[i] = src[i] * inv * (weight[i] + 1.0);
    }
}

/// LayerNorm (elementwise_affine=False) fused with AdaLN modulation: `LN(x) * (1 + scale)`
#[inline]
pub fn layer_norm_modulate(src: &[f32], dst: &mut [f32], scale: &[f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                layer_norm_modulate_avx2(src, dst, scale, eps);
            }
            return;
        }
    }
    let n = src.len();
    let mut sum = 0.0f32;
    for &v in src {
        sum += v;
    }
    let mean = sum / (n as f32);
    let mut var = 0.0f32;
    for &v in src {
        let d = v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / (n as f32) + eps).sqrt();
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv * (1.0 + scale[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_modulate_avx2(src: &[f32], dst: &mut [f32], scale: &[f32], eps: f32) {
    unsafe {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let scp = scale.as_ptr();

        let mut vsum = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(sp.add(i)));
            i += 8;
        }
        let hsum = |v: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let s = _mm_add_ps(lo, hi);
            let sh = _mm_movehdup_ps(s);
            let ss = _mm_add_ps(s, sh);
            _mm_cvtss_f32(_mm_add_ss(ss, _mm_movehl_ps(sh, ss)))
        };
        let mut sum = hsum(vsum);
        while i < n {
            sum += *sp.add(i);
            i += 1;
        }
        let mean = sum / (n as f32);
        let vmean = _mm256_set1_ps(mean);

        let mut vvar = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(sp.add(i)), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
            i += 8;
        }
        let mut var = hsum(vvar);
        while i < n {
            let d = *sp.add(i) - mean;
            var += d * d;
            i += 1;
        }
        let inv = 1.0 / (var / (n as f32) + eps).sqrt();
        let vinv = _mm256_set1_ps(inv);
        let vone = _mm256_set1_ps(1.0);
        let mut i = 0;
        while i + 8 <= n {
            let norm = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(sp.add(i)), vmean), vinv);
            let mod_s = _mm256_add_ps(vone, _mm256_loadu_ps(scp.add(i)));
            _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(norm, mod_s));
            i += 8;
        }
        while i < n {
            *dp.add(i) = (*sp.add(i) - mean) * inv * (1.0 + *scp.add(i));
            i += 1;
        }
    }
}

/// Vectorized in-place SiLU: `x[i] = x[i] / (1 + exp(-x[i]))`
#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                silu_inplace_avx2(x);
            }
            return;
        }
    }
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_inplace_avx2(x: &mut [f32]) {
    unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let neg_zero = _mm256_set1_ps(-0.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let e = exp256_ps(_mm256_xor_ps(v, neg_zero));
            _mm256_storeu_ps(p.add(i), _mm256_div_ps(v, _mm256_add_ps(one, e)));
            i += 8;
        }
        while i < n {
            *p.add(i) = silu(*p.add(i));
            i += 1;
        }
    }
}

/// Vectorized gated residual: `x[i] += gate[i] * delta[i]`
#[inline]
pub fn gated_residual_add(x: &mut [f32], delta: &[f32], gate: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                gated_residual_add_avx2(x, delta, gate);
            }
            return;
        }
    }
    for i in 0..x.len() {
        x[i] += gate[i] * delta[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gated_residual_add_avx2(x: &mut [f32], delta: &[f32], gate: &[f32]) {
    unsafe {
        let n = x.len();
        let xp = x.as_mut_ptr();
        let dp = delta.as_ptr();
        let gp = gate.as_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(xp.add(i));
            let dv = _mm256_loadu_ps(dp.add(i));
            let gv = _mm256_loadu_ps(gp.add(i));
            _mm256_storeu_ps(xp.add(i), _mm256_fmadd_ps(gv, dv, xv));
            i += 8;
        }
        while i < n {
            *xp.add(i) += *gp.add(i) * *dp.add(i);
            i += 1;
        }
    }
}

/// Pack `K_head = [k_txt (L_t x 128); k_img (L_i x 128)]` into `Q8K128` (`k_groups = 1`).
pub fn pack_attn_k_q8(k_txt: &[f32], k_img: &[f32]) -> Vec<u8> {
    let l_t = k_txt.len() / GROUP_SIZE;
    let l_i = k_img.len() / GROUP_SIZE;
    let total = l_t + l_i;
    let n_tiles = align_up(total, NR_TILE) / NR_TILE;
    let mut out = vec![0u8; n_tiles * Q8K128_BLOCK_BYTES];

    for nt in 0..n_tiles {
        let n0 = nt * NR_TILE;
        let block = &mut out[nt * Q8K128_BLOCK_BYTES..(nt + 1) * Q8K128_BLOCK_BYTES];
        let (w_bytes, rest) = block.split_at_mut(2048);
        let (bias_bytes, scale_bytes) = rest.split_at_mut(64);
        let biases: &mut [i32] = bytemuck::cast_slice_mut(bias_bytes);
        let scales: &mut [f32] = bytemuck::cast_slice_mut(scale_bytes);

        for c in 0..NR_TILE {
            let idx = n0 + c;
            if idx >= total {
                scales[c] = 0.0;
                biases[c] = 0;
                continue;
            }
            let row = if idx < l_t {
                &k_txt[idx * GROUP_SIZE..(idx + 1) * GROUP_SIZE]
            } else {
                let ii = idx - l_t;
                &k_img[ii * GROUP_SIZE..(ii + 1) * GROUP_SIZE]
            };
            let mut max_abs = 0.0f32;
            for &v in row {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales[c] = scale;

            let mut sum_q = 0i32;
            let half_off = if c < 8 { c * 4 } else { 32 + (c - 8) * 4 };
            for k4 in 0..32 {
                let base = k4 * 64 + half_off;
                for r in 0..4 {
                    let q = (row[k4 * 4 + r] * inv).round().clamp(-127.0, 127.0) as i8;
                    sum_q += q as i32;
                    w_bytes[base + r] = q as u8;
                }
            }
            biases[c] = sum_q * 128;
        }
    }
    out
}

/// Pack `V_head^T` (`N = 128` output dims, `K = L_t + L_i` keys) into `Q8K128` with `w_bias = 0`
/// (because softmax weights `P` are non-negative `u8` with zero_point = 0!).
/// Optimized row-wise traversal so `v_txt` and `v_img` are read sequentially in cache lines!
pub fn pack_attn_v_q8(v_txt: &[f32], v_img: &[f32]) -> Vec<u8> {
    let l_t = v_txt.len() / GROUP_SIZE;
    let l_i = v_img.len() / GROUP_SIZE;
    let total = l_t + l_i;
    let k_pad = align_up(total, GROUP_SIZE);
    let k_groups = k_pad / GROUP_SIZE;
    let n_tiles = GROUP_SIZE / NR_TILE; // 8
    let mut out = vec![0u8; n_tiles * k_groups * Q8K128_BLOCK_BYTES];

    let mut max_abs = [0.0f32; GROUP_SIZE];
    let mut inv_scales = [0.0f32; GROUP_SIZE];

    for g in 0..k_groups {
        let s0 = g * GROUP_SIZE;
        let s_end = (s0 + GROUP_SIZE).min(total);
        max_abs.fill(0.0);

        // Pass 1: row-wise max_abs over all 128 channels simultaneously (sequential memory access!)
        for s in s0..s_end {
            let row = if s < l_t {
                &v_txt[s * GROUP_SIZE..(s + 1) * GROUP_SIZE]
            } else {
                &v_img[(s - l_t) * GROUP_SIZE..(s - l_t + 1) * GROUP_SIZE]
            };
            for d in 0..GROUP_SIZE {
                let a = row[d].abs();
                if a > max_abs[d] {
                    max_abs[d] = a;
                }
            }
        }

        for d in 0..GROUP_SIZE {
            let sc = if max_abs[d] > 0.0 {
                max_abs[d] / 127.0
            } else {
                0.0
            };
            inv_scales[d] = if sc > 0.0 { 1.0 / sc } else { 0.0 };
            let nt = d / NR_TILE;
            let c = d % NR_TILE;
            let scale_off = (nt * k_groups + g) * Q8K128_BLOCK_BYTES + 2112 + c * 4;
            out[scale_off..scale_off + 4].copy_from_slice(&sc.to_le_bytes());
        }

        // Pass 2: row-wise quantize and scatter into the 8 N-tiles
        for s in s0..s_end {
            let row = if s < l_t {
                &v_txt[s * GROUP_SIZE..(s + 1) * GROUP_SIZE]
            } else {
                &v_img[(s - l_t) * GROUP_SIZE..(s - l_t + 1) * GROUP_SIZE]
            };
            let rel = s - s0;
            let k4 = rel >> 2;
            let r = rel & 3;
            for nt in 0..n_tiles {
                let d0 = nt * NR_TILE;
                let block_base = (nt * k_groups + g) * Q8K128_BLOCK_BYTES + k4 * 64;
                for c in 0..8 {
                    let d = d0 + c;
                    let q = (row[d] * inv_scales[d]).round().clamp(-127.0, 127.0) as i8;
                    out[block_base + c * 4 + r] = q as u8;
                }
                for c in 0..8 {
                    let d = d0 + 8 + c;
                    let q = (row[d] * inv_scales[d]).round().clamp(-127.0, 127.0) as i8;
                    out[block_base + 32 + c * 4 + r] = q as u8;
                }
            }
        }
    }
    out
}

/// Quantize a non-negative softmax probability row `probs[..len]` per 128-key group to `u8` `[0, 255]`
/// with `zero_point = 0` using AVX2.
#[inline]
fn quantize_probs_u8_k128(probs: &[f32], qrow: &mut [u8], srow: &mut [f32], k_groups: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                quantize_probs_u8_k128_avx2(probs, qrow, srow, k_groups);
            }
            return;
        }
    }
    let len = probs.len();
    for g in 0..k_groups {
        let s0 = g * GROUP_SIZE;
        let s_end = (s0 + GROUP_SIZE).min(len);
        let mut p_max = 0.0f32;
        for &v in &probs[s0..s_end] {
            if v > p_max {
                p_max = v;
            }
        }
        let scale = if p_max > 0.0 { p_max / 255.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        srow[g] = scale;
        for j in 0..GROUP_SIZE {
            let idx = s0 + j;
            if idx < len {
                qrow[idx] = (probs[idx] * inv).round().clamp(0.0, 255.0) as u8;
            } else {
                qrow[idx] = 0;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_probs_u8_k128_avx2(
    probs: &[f32],
    qrow: &mut [u8],
    srow: &mut [f32],
    k_groups: usize,
) {
    unsafe {
        let len = probs.len();
        let v_zero = _mm256_setzero_ps();
        let v_255 = _mm256_set1_ps(255.0);
        let perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        for g in 0..k_groups {
            let s0 = g * GROUP_SIZE;
            if s0 + GROUP_SIZE <= len {
                let sp = probs.as_ptr().add(s0);
                let mut vmax0 = _mm256_setzero_ps();
                let mut vmax1 = _mm256_setzero_ps();
                let mut j = 0;
                while j < GROUP_SIZE {
                    vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(sp.add(j)));
                    vmax1 = _mm256_max_ps(vmax1, _mm256_loadu_ps(sp.add(j + 8)));
                    j += 16;
                }
                let vmax = _mm256_max_ps(vmax0, vmax1);
                let hi = _mm256_extractf128_ps(vmax, 1);
                let lo = _mm256_castps256_ps128(vmax);
                let m128 = _mm_max_ps(lo, hi);
                let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
                let p_max = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1)));

                let scale = if p_max > 0.0 { p_max / 255.0 } else { 0.0 };
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                *srow.get_unchecked_mut(g) = scale;

                let vinv = _mm256_set1_ps(inv);
                let dp = qrow.as_mut_ptr().add(s0);
                let mut j = 0;
                while j < GROUP_SIZE {
                    let f0 = _mm256_min_ps(
                        _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(sp.add(j)), vinv), v_zero),
                        v_255,
                    );
                    let f1 = _mm256_min_ps(
                        _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 8)), vinv), v_zero),
                        v_255,
                    );
                    let f2 = _mm256_min_ps(
                        _mm256_max_ps(
                            _mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 16)), vinv),
                            v_zero,
                        ),
                        v_255,
                    );
                    let f3 = _mm256_min_ps(
                        _mm256_max_ps(
                            _mm256_mul_ps(_mm256_loadu_ps(sp.add(j + 24)), vinv),
                            v_zero,
                        ),
                        v_255,
                    );
                    let p01 = _mm256_packus_epi32(_mm256_cvtps_epi32(f0), _mm256_cvtps_epi32(f1));
                    let p23 = _mm256_packus_epi32(_mm256_cvtps_epi32(f2), _mm256_cvtps_epi32(f3));
                    let p0123 = _mm256_packus_epi16(p01, p23);
                    let ordered = _mm256_permutevar8x32_epi32(p0123, perm_idx);
                    _mm256_storeu_si256(dp.add(j) as *mut __m256i, ordered);
                    j += 32;
                }
            } else {
                let s_end = len;
                let mut p_max = 0.0f32;
                for &v in &probs[s0..s_end] {
                    if v > p_max {
                        p_max = v;
                    }
                }
                let scale = if p_max > 0.0 { p_max / 255.0 } else { 0.0 };
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                *srow.get_unchecked_mut(g) = scale;
                for j in 0..GROUP_SIZE {
                    let idx = s0 + j;
                    if idx < len {
                        *qrow.get_unchecked_mut(idx) =
                            (probs[idx] * inv).round().clamp(0.0, 255.0) as u8;
                    } else {
                        *qrow.get_unchecked_mut(idx) = 0;
                    }
                }
            }
        }
    }
}

/// Single-thread AVX-VNNI SageAttention tile for `q_chunk` queries (`[qm, 128]`) of ONE head,
/// writing `[qm, 128]` into `out_cb(local_q, &[f32; 128])` or contiguous `out_chunk`.
pub fn sdpa_vnni_qchunk(
    q_chunk: &[f32],
    qm: usize,
    k_packed: &[u8],
    v_packed: &[u8],
    total_kv: usize,
    attn_scale: f32,
    out_chunk: &mut [f32],
) {
    debug_assert_eq!(q_chunk.len(), qm * GROUP_SIZE);
    debug_assert_eq!(out_chunk.len(), qm * GROUP_SIZE);

    let kv_tiles = align_up(total_kv, NR_TILE) / NR_TILE;
    let kv_pad = align_up(total_kv, GROUP_SIZE);
    let kv_groups = kv_pad / GROUP_SIZE;

    // 1. Quantize Q chunk [qm, 128] -> q_u8 [qm, 128], q_scales [qm] (scaled by attn_scale!)
    let mut q_u8 = vec![128u8; qm * GROUP_SIZE];
    let mut q_scales = vec![0.0f32; qm];
    for r in 0..qm {
        let src = &q_chunk[r * GROUP_SIZE..(r + 1) * GROUP_SIZE];
        let qrow = &mut q_u8[r * GROUP_SIZE..(r + 1) * GROUP_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    quantize_row_avx2(src, qrow, &mut q_scales[r..r + 1], GROUP_SIZE, 1);
                }
                q_scales[r] *= attn_scale;
                continue;
            }
        }
        quantize_row_scalar(src, qrow, &mut q_scales[r..r + 1], GROUP_SIZE, 1);
        q_scales[r] *= attn_scale;
    }

    // 2. Scores [qm, total_kv] = Q_u8 @ K_packed^T (k_groups = 1!)
    let mut scores = vec![0.0f32; qm * total_kv];
    #[cfg(target_arch = "x86_64")]
    let use_vnni = std::arch::is_x86_feature_detected!("avxvnni");
    #[cfg(not(target_arch = "x86_64"))]
    let use_vnni = false;

    for nt in 0..kv_tiles {
        let n0 = nt * NR_TILE;
        let n_valid = (total_kv - n0).min(NR_TILE);
        let w_tile = unsafe { k_packed.as_ptr().add(nt * Q8K128_BLOCK_BYTES) };
        let mut r = 0;
        while r + 6 <= qm {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if use_vnni {
                    microkernel_q8_6x16_vnni_1g(
                        q_u8.as_ptr().add(r * GROUP_SIZE),
                        q_scales.as_ptr().add(r),
                        w_tile,
                        scores.as_mut_ptr().add(r * total_kv + n0),
                        total_kv,
                        n_valid,
                    );
                } else {
                    microkernel_q8_scalar(
                        q_u8.as_ptr().add(r * GROUP_SIZE),
                        q_scales.as_ptr().add(r),
                        w_tile,
                        scores.as_mut_ptr().add(r * total_kv + n0),
                        None,
                        6,
                        GROUP_SIZE,
                        1,
                        total_kv,
                        n_valid,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                microkernel_q8_scalar(
                    q_u8.as_ptr().add(r * GROUP_SIZE),
                    q_scales.as_ptr().add(r),
                    w_tile,
                    scores.as_mut_ptr().add(r * total_kv + n0),
                    None,
                    6,
                    GROUP_SIZE,
                    1,
                    total_kv,
                    n_valid,
                );
            }
            r += 6;
        }
        let rem = qm - r;
        if rem > 0 {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if use_vnni {
                    microkernel_q8_remx16_vnni(
                        q_u8.as_ptr().add(r * GROUP_SIZE),
                        q_scales.as_ptr().add(r),
                        w_tile,
                        scores.as_mut_ptr().add(r * total_kv + n0),
                        None,
                        rem,
                        GROUP_SIZE,
                        1,
                        total_kv,
                        n_valid,
                    );
                } else {
                    microkernel_q8_scalar(
                        q_u8.as_ptr().add(r * GROUP_SIZE),
                        q_scales.as_ptr().add(r),
                        w_tile,
                        scores.as_mut_ptr().add(r * total_kv + n0),
                        None,
                        rem,
                        GROUP_SIZE,
                        1,
                        total_kv,
                        n_valid,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                microkernel_q8_scalar(
                    q_u8.as_ptr().add(r * GROUP_SIZE),
                    q_scales.as_ptr().add(r),
                    w_tile,
                    scores.as_mut_ptr().add(r * total_kv + n0),
                    None,
                    rem,
                    GROUP_SIZE,
                    1,
                    total_kv,
                    n_valid,
                );
            }
        }
    }

    // 3. Softmax + quantize P [qm, kv_pad] -> u8 [0..255]
    let mut p_u8 = vec![0u8; qm * kv_pad];
    let mut p_scales = vec![0.0f32; qm * kv_groups];
    for r in 0..qm {
        let srow = &mut scores[r * total_kv..(r + 1) * total_kv];
        softmax_inplace(srow);
        quantize_probs_u8_k128(
            srow,
            &mut p_u8[r * kv_pad..(r + 1) * kv_pad],
            &mut p_scales[r * kv_groups..(r + 1) * kv_groups],
            kv_groups,
        );
    }

    // 4. Out [qm, 128] = P_u8 @ V_packed^T (8 N-tiles of 16, w_bias = 0!)
    let v_tile_stride = kv_groups * Q8K128_BLOCK_BYTES;
    for nt in 0..(GROUP_SIZE / NR_TILE) {
        let n0 = nt * NR_TILE;
        let w_tile = unsafe { v_packed.as_ptr().add(nt * v_tile_stride) };
        let mut r = 0;
        while r + 6 <= qm {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if use_vnni {
                    microkernel_q8_6x16_vnni(
                        p_u8.as_ptr().add(r * kv_pad),
                        p_scales.as_ptr().add(r * kv_groups),
                        w_tile,
                        out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                        None,
                        kv_pad,
                        kv_groups,
                        GROUP_SIZE,
                        NR_TILE,
                    );
                } else {
                    microkernel_q8_scalar(
                        p_u8.as_ptr().add(r * kv_pad),
                        p_scales.as_ptr().add(r * kv_groups),
                        w_tile,
                        out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                        None,
                        6,
                        kv_pad,
                        kv_groups,
                        GROUP_SIZE,
                        NR_TILE,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                microkernel_q8_scalar(
                    p_u8.as_ptr().add(r * kv_pad),
                    p_scales.as_ptr().add(r * kv_groups),
                    w_tile,
                    out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                    None,
                    6,
                    kv_pad,
                    kv_groups,
                    GROUP_SIZE,
                    NR_TILE,
                );
            }
            r += 6;
        }
        let rem = qm - r;
        if rem > 0 {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if use_vnni {
                    microkernel_q8_remx16_vnni(
                        p_u8.as_ptr().add(r * kv_pad),
                        p_scales.as_ptr().add(r * kv_groups),
                        w_tile,
                        out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                        None,
                        rem,
                        kv_pad,
                        kv_groups,
                        GROUP_SIZE,
                        NR_TILE,
                    );
                } else {
                    microkernel_q8_scalar(
                        p_u8.as_ptr().add(r * kv_pad),
                        p_scales.as_ptr().add(r * kv_groups),
                        w_tile,
                        out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                        None,
                        rem,
                        kv_pad,
                        kv_groups,
                        GROUP_SIZE,
                        NR_TILE,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                microkernel_q8_scalar(
                    p_u8.as_ptr().add(r * kv_pad),
                    p_scales.as_ptr().add(r * kv_groups),
                    w_tile,
                    out_chunk.as_mut_ptr().add(r * GROUP_SIZE + n0),
                    None,
                    rem,
                    kv_pad,
                    kv_groups,
                    GROUP_SIZE,
                    NR_TILE,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn test_softmax_and_swiglu_accuracy() {
        let mut rng = Rng::from_seed(123);
        let mut s = vec![0.0f32; 1043];
        rng.fill_randn(&mut s);
        let mut s_ref = s.clone();
        // scalar reference
        let m = s_ref.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &mut s_ref {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in &mut s_ref {
            *v /= sum;
        }

        softmax_inplace(&mut s);
        let max_err = s
            .iter()
            .zip(s_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "softmax max_err={max_err}");

        let mut g = vec![0.0f32; 3456];
        let mut u = vec![0.0f32; 3456];
        rng.fill_randn(&mut g);
        rng.fill_randn(&mut u);
        let mut out = vec![0.0f32; 3456];
        swiglu_row(&g, &u, &mut out);
        let max_err_sw = out
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - silu(g[i]) * u[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err_sw < 1e-5, "swiglu max_err={max_err_sw}");
    }

    #[test]
    fn test_q8_and_q4_gemm_tail_shapes() {
        let mut rng = Rng::from_seed(77);
        // Non-multiple-of-6 M, non-multiple-of-16 N, non-multiple-of-128 K
        let (m, n, k) = (19, 37, 200);
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        let mut bias = vec![0.0f32; n];
        rng.fill_randn(&mut a);
        rng.fill_randn(&mut b);
        rng.fill_randn(&mut bias);

        let mut ref_out = vec![0.0f32; m * n];
        matmul_f32(&a, m, k, &b, n, Some(&bias), &mut ref_out);

        let q8 = pack_q8k128(&b, n, k);
        let w8 = QuantMatrixView { dtype: TensorDType::Q8K128, n, k, data: &q8 };
        let mut out8 = vec![0.0f32; m * n];
        let mut buf = ActQuantBuf::default();
        linear(&a, m, w8, Some(&bias), &mut out8, &mut buf);

        let q4 = pack_q4k128(&b, n, k);
        let w4 = QuantMatrixView { dtype: TensorDType::Q4K128, n, k, data: &q4 };
        let mut out4 = vec![0.0f32; m * n];
        linear(&a, m, w4, Some(&bias), &mut out4, &mut buf);

        let cos = |p: &[f32]| -> f64 {
            let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
            for i in 0..p.len() {
                d += p[i] as f64 * ref_out[i] as f64;
                na += p[i] as f64 * p[i] as f64;
                nb += ref_out[i] as f64 * ref_out[i] as f64;
            }
            d / (na.sqrt() * nb.sqrt())
        };
        assert!(cos(&out8) > 0.9998, "Q8 cos={}", cos(&out8));
        assert!(cos(&out4) > 0.990, "Q4 cos={}", cos(&out4));
    }
}
