//! Micro-benchmark & numerical accuracy verification for Q8K128 / Q4K128 AVX-VNNI GEMM kernels.

use haidass_image_rs::format::{QuantMatrixView, TensorDType};
use haidass_image_rs::rng::Rng;
use haidass_image_rs::simd::{
    axpy4_f32, dot4_f32, linear, matmul_f32, pack_attn_k_q8, pack_attn_v_q8, pack_q4k128,
    pack_q8k128, sdpa_vnni_qchunk, softmax_inplace, swiglu_row, ActQuantBuf,
};
use rayon::prelude::*;
use std::time::Instant;

fn bench_shape(m: usize, n: usize, k: usize, iters: usize) {
    let mut rng = Rng::from_seed(42);
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; n * k];
    rng.fill_randn(&mut a);
    rng.fill_randn(&mut b);

    let q8_bytes = pack_q8k128(&b, n, k);
    let q4_bytes = pack_q4k128(&b, n, k);
    let w_q8 = QuantMatrixView {
        dtype: TensorDType::Q8K128,
        n,
        k,
        data: &q8_bytes,
    };
    let w_q4 = QuantMatrixView {
        dtype: TensorDType::Q4K128,
        n,
        k,
        data: &q4_bytes,
    };

    let mut out_q8 = vec![0.0f32; m * n];
    let mut out_q4 = vec![0.0f32; m * n];
    let mut scratch = ActQuantBuf::default();

    // Warmup
    linear(&a, m, w_q8, None, &mut out_q8, &mut scratch);
    linear(&a, m, w_q4, None, &mut out_q4, &mut scratch);

    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);

    let t0 = Instant::now();
    for _ in 0..iters {
        linear(&a, m, w_q8, None, &mut out_q8, &mut scratch);
    }
    let dt_q8 = t0.elapsed().as_secs_f64() / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters {
        linear(&a, m, w_q4, None, &mut out_q4, &mut scratch);
    }
    let dt_q4 = t1.elapsed().as_secs_f64() / (iters as f64);

    // Check accuracy on first 16 rows against exact F32
    let m_check = m.min(16);
    let mut out_ref = vec![0.0f32; m_check * n];
    matmul_f32(&a[..m_check * k], m_check, k, &b, n, None, &mut out_ref);
    let cosine = |pred: &[f32]| -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..out_ref.len() {
            let x = out_ref[i] as f64;
            let y = pred[i] as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    };

    println!(
        "GEMM [{:4} x {:4} x {:4}] | Q8: {:6.2} ms ({:5.2} TOPS, cos={:.6}) | Q4: {:6.2} ms ({:5.2} TOPS, cos={:.5})",
        m,
        n,
        k,
        dt_q8 * 1000.0,
        (ops / dt_q8) / 1e12,
        cosine(&out_q8[..m_check * n]),
        dt_q4 * 1000.0,
        (ops / dt_q4) / 1e12,
        cosine(&out_q4[..m_check * n]),
    );
}

fn main() {
    let threads = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .unwrap();
    }
    println!(
        "haidass-bench | threads={} | avxvnni={}",
        rayon::current_num_threads(),
        std::arch::is_x86_feature_detected!("avxvnni")
    );
    // Exact DiT CFG-batched GEMM shapes!
    bench_shape(2048, 3456, 1152, 8); // Fused QKV
    bench_shape(2048, 1152, 1152, 10); // O proj
    bench_shape(2048, 6912, 1152, 6); // Fused GateUp
    bench_shape(2048, 1152, 3456, 8); // MLP Down

    // Benchmark 4Q-blocked SDPA for 18 heads (CFG=2 x 9 heads, 1024 queries x 1056 keys x 128)
    let mut rng = Rng::from_seed(99);
    let heads = 18usize;
    let cells = 1024usize;
    let kv_len = 1056usize;
    let d = 128usize;
    let mut q = vec![0.0f32; heads * cells * d];
    let mut k = vec![0.0f32; heads * kv_len * d];
    let mut v = vec![0.0f32; heads * kv_len * d];
    let mut out = vec![0.0f32; heads * cells * d];
    rng.fill_randn(&mut q);
    rng.fill_randn(&mut k);
    rng.fill_randn(&mut v);
    let scale = 1.0 / (d as f32).sqrt();
    let q_tile = 64usize;
    let q_blocks = cells / q_tile;

    let run_sdpa = |out: &mut [f32]| {
        let out_ptr = out.as_mut_ptr() as usize;
        (0..heads * q_blocks).into_par_iter().for_each(|task| {
            let h = task / q_blocks;
            let qb = task % q_blocks;
            let q0 = qb * q_tile;
            let q1 = q0 + q_tile;
            let qh = &q[h * cells * d..(h + 1) * cells * d];
            let kh = &k[h * kv_len * d..(h + 1) * kv_len * d];
            let vh = &v[h * kv_len * d..(h + 1) * kv_len * d];
            let mut sc0 = vec![0.0f32; kv_len];
            let mut sc1 = vec![0.0f32; kv_len];
            let mut sc2 = vec![0.0f32; kv_len];
            let mut sc3 = vec![0.0f32; kv_len];
            let mut a0 = [0.0f32; 128];
            let mut a1 = [0.0f32; 128];
            let mut a2 = [0.0f32; 128];
            let mut a3 = [0.0f32; 128];
            let mut qi = q0;
            while qi + 4 <= q1 {
                let qv0 = &qh[qi * d..(qi + 1) * d];
                let qv1 = &qh[(qi + 1) * d..(qi + 2) * d];
                let qv2 = &qh[(qi + 2) * d..(qi + 3) * d];
                let qv3 = &qh[(qi + 3) * d..(qi + 4) * d];
                for s in 0..kv_len {
                    let dd = dot4_f32(qv0, qv1, qv2, qv3, &kh[s * d..(s + 1) * d]);
                    sc0[s] = dd[0] * scale;
                    sc1[s] = dd[1] * scale;
                    sc2[s] = dd[2] * scale;
                    sc3[s] = dd[3] * scale;
                }
                softmax_inplace(&mut sc0);
                softmax_inplace(&mut sc1);
                softmax_inplace(&mut sc2);
                softmax_inplace(&mut sc3);
                a0.fill(0.0);
                a1.fill(0.0);
                a2.fill(0.0);
                a3.fill(0.0);
                for s in 0..kv_len {
                    axpy4_f32(&mut a0, &mut a1, &mut a2, &mut a3, &vh[s * d..(s + 1) * d], [sc0[s], sc1[s], sc2[s], sc3[s]]);
                }
                unsafe {
                    let base = (out_ptr as *mut f32).add(h * cells * d);
                    std::ptr::copy_nonoverlapping(a0.as_ptr(), base.add(qi * d), d);
                    std::ptr::copy_nonoverlapping(a1.as_ptr(), base.add((qi + 1) * d), d);
                    std::ptr::copy_nonoverlapping(a2.as_ptr(), base.add((qi + 2) * d), d);
                    std::ptr::copy_nonoverlapping(a3.as_ptr(), base.add((qi + 3) * d), d);
                }
                qi += 4;
            }
        });
    };
    run_sdpa(&mut out);
    let ts = Instant::now();
    for _ in 0..5 {
        run_sdpa(&mut out);
    }
    let sdpa_ms = ts.elapsed().as_secs_f64() * 1000.0 / 5.0;

    // Benchmark AVX-VNNI SageAttention vs F32 SDPA!
    let mut out_vnni = vec![0.0f32; heads * cells * d];
    let run_sdpa_vnni = |out_v: &mut [f32]| {
        let packed: Vec<(Vec<u8>, Vec<u8>)> = (0..heads)
            .into_par_iter()
            .map(|h| {
                let kh = &k[h * kv_len * d..(h + 1) * kv_len * d];
                let vh = &v[h * kv_len * d..(h + 1) * kv_len * d];
                (pack_attn_k_q8(&[], kh), pack_attn_v_q8(&[], vh))
            })
            .collect();
        let q_tile_v = 48usize;
        let q_blocks_v = (cells + q_tile_v - 1) / q_tile_v;
        let out_ptr = out_v.as_mut_ptr() as usize;
        (0..heads * q_blocks_v).into_par_iter().for_each(|task| {
            let h = task / q_blocks_v;
            let qb = task % q_blocks_v;
            let q0 = qb * q_tile_v;
            let qm = (cells - q0).min(q_tile_v);
            let qh = &q[(h * cells + q0) * d..(h * cells + q0 + qm) * d];
            let mut chunk = vec![0.0f32; qm * d];
            sdpa_vnni_qchunk(qh, qm, &packed[h].0, &packed[h].1, kv_len, scale, &mut chunk);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    (out_ptr as *mut f32).add((h * cells + q0) * d),
                    qm * d,
                );
            }
        });
    };
    run_sdpa_vnni(&mut out_vnni);
    let tv = Instant::now();
    for _ in 0..8 {
        run_sdpa_vnni(&mut out_vnni);
    }
    let sdpa_vnni_ms = tv.elapsed().as_secs_f64() * 1000.0 / 8.0;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..out.len() {
        dot += out[i] as f64 * out_vnni[i] as f64;
        na += out[i] as f64 * out[i] as f64;
        nb += out_vnni[i] as f64 * out_vnni[i] as f64;
    }
    let sdpa_cos = dot / (na.sqrt() * nb.sqrt());

    // Benchmark SwiGLU [2048, 3456]
    let mut gu = vec![0.0f32; 2048 * 6912];
    let mut mid = vec![0.0f32; 2048 * 3456];
    rng.fill_randn(&mut gu);
    let tsw = Instant::now();
    for _ in 0..10 {
        mid.par_chunks_mut(3456).enumerate().for_each(|(i, dst)| {
            let row = &gu[i * 6912..(i + 1) * 6912];
            swiglu_row(&row[..3456], &row[3456..], dst);
        });
    }
    let swiglu_ms = tsw.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!(
        "SDPA F32  18H: {:5.2} ms/layer ({:.2} s/step)\nSDPA VNNI 18H: {:5.2} ms/layer ({:.2} s/step, cos={:.6}) | SwiGLU: {:4.2} ms/layer",
        sdpa_ms,
        sdpa_ms * 34.0 / 1000.0,
        sdpa_vnni_ms,
        sdpa_vnni_ms * 34.0 / 1000.0,
        sdpa_cos,
        swiglu_ms,
    );
}
