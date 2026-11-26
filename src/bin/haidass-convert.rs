//! Convert Hugging Face `DALabCommunity/Haidass-Image-Instruct` + `Qwen/Qwen3-0.6B` safetensors
//! into packed, quantized, 64-byte aligned `.himg` files for ultra-fast AVX-VNNI CPU inference.

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use half::bf16;
use haidass_image_rs::dit::DIT_LAYERS;
use haidass_image_rs::format::{StreamingHimgWriter, TensorDType};
use haidass_image_rs::qwen3::QWEN3_LAYERS;
use haidass_image_rs::simd::{pack_emb_q8, pack_q4k128, pack_q8k128};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantMode {
    /// W8A8 (Q8K128) for DiT, Qwen3, and VAE heavy convs (~1.35 GB, near-lossless)
    Q8,
    /// W4A8 (Q4K128) for DiT & Qwen3, Q8K128 for VAE (~0.80 GB, best speed/size/quality balance)
    Q4Mixed,
    /// W4A8 (Q4K128) everywhere (~0.70 GB, smallest footprint)
    Q4,
    /// Pure F32 (unquantized reference for accuracy comparison)
    F32,
}

impl QuantMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Q4Mixed => "q4_mixed",
            Self::Q4 => "q4",
            Self::F32 => "f32",
        }
    }

    fn dit_dtype(&self) -> TensorDType {
        match self {
            Self::Q8 => TensorDType::Q8K128,
            Self::Q4Mixed | Self::Q4 => TensorDType::Q4K128,
            Self::F32 => TensorDType::F32,
        }
    }

    fn qwen3_dtype(&self) -> TensorDType {
        match self {
            Self::Q8 => TensorDType::Q8K128,
            Self::Q4Mixed | Self::Q4 => TensorDType::Q4K128,
            Self::F32 => TensorDType::F32,
        }
    }

    fn vae_dtype(&self) -> TensorDType {
        match self {
            Self::Q8 | Self::Q4Mixed => TensorDType::Q8K128,
            Self::Q4 => TensorDType::Q4K128,
            Self::F32 => TensorDType::F32,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "haidass-convert", about = "Convert & quantize Haidass-Image-Instruct + Qwen3-0.6B into .himg")]
struct Args {
    /// Local path to DALabCommunity/Haidass-Image-Instruct directory
    #[arg(long)]
    model_dir: PathBuf,

    /// Local path to Qwen/Qwen3-0.6B directory
    #[arg(long)]
    text_encoder_dir: PathBuf,

    /// Output path for the main quantized .himg bundle (Qwen3 + DiT Base + VAE Decoder)
    #[arg(short, long)]
    output: PathBuf,

    /// Optional output path for pre-merged writing-LoRA DiT .himg (zero runtime LoRA overhead)
    #[arg(long)]
    writing_output: Option<PathBuf>,

    /// Quantization mode
    #[arg(long, value_enum, default_value_t = QuantMode::Q8)]
    quant: QuantMode,
}

fn load_tensor_f32(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = st
        .tensor(name)
        .with_context(|| format!("Missing tensor '{name}' in safetensors"))?;
    let shape = view.shape().to_vec();
    let data = view.data();
    let vals = match view.dtype() {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
        Dtype::BF16 => {
            let raw: &[u16] = bytemuck::cast_slice(data);
            raw.par_iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect()
        }
        Dtype::F16 => {
            let raw: &[u16] = bytemuck::cast_slice(data);
            raw.par_iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        }
        other => bail!("Unsupported safetensors dtype {:?} for '{name}'", other),
    };
    Ok((shape, vals))
}

fn write_linear(
    writer: &mut StreamingHimgWriter,
    name: &str,
    data: &[f32],
    n: usize,
    k: usize,
    dtype: TensorDType,
) -> Result<()> {
    match dtype {
        TensorDType::Q8K128 => {
            let packed = pack_q8k128(data, n, k);
            writer.add_raw(name, TensorDType::Q8K128, vec![n, k], &packed)
        }
        TensorDType::Q4K128 => {
            let packed = pack_q4k128(data, n, k);
            writer.add_raw(name, TensorDType::Q4K128, vec![n, k], &packed)
        }
        TensorDType::F32 => writer.add_f32(name, vec![n, k], data),
        other => bail!("Unexpected linear dtype {:?}", other),
    }
}

/// Convert PyTorch Conv2d weight `[C_out, C_in, kH, kW]` into HWC-im2col matrix `[C_out, kH*kW*C_in]`.
fn transpose_conv_oihw_to_ohwi(w: &[f32], c_out: usize, c_in: usize, kh: usize, kw: usize) -> Vec<f32> {
    let k_spatial = kh * kw;
    let k_dim = k_spatial * c_in;
    let mut out = vec![0.0f32; c_out * k_dim];
    out.par_chunks_mut(k_dim).enumerate().for_each(|(oc, dst)| {
        for ic in 0..c_in {
            for s in 0..k_spatial {
                let src_idx = (oc * c_in + ic) * k_spatial + s;
                dst[s * c_in + ic] = w[src_idx];
            }
        }
    });
    out
}

/// Merge LoRA: `W += scale * (B @ A)` where `W` is `[out, in]`, `B` is `[out, r]`, `A` is `[r, in]`.
fn merge_lora_inplace(w: &mut [f32], b: &[f32], a: &[f32], out_dim: usize, in_dim: usize, rank: usize, scale: f32) {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(b.len(), out_dim * rank);
    assert_eq!(a.len(), rank * in_dim);
    w.par_chunks_mut(in_dim).enumerate().for_each(|(o, wrow)| {
        let brow = &b[o * rank..(o + 1) * rank];
        for r in 0..rank {
            let coeff = scale * brow[r];
            let arow = &a[r * in_dim..(r + 1) * in_dim];
            for i in 0..in_dim {
                wrow[i] += coeff * arow[i];
            }
        }
    });
}

fn load_norm_remainders(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let mut map = HashMap::new();
    if !path.is_file() {
        return Ok(map);
    }
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut bases = std::collections::HashSet::new();
    for name in st.names() {
        if let Some((base, _)) = name.rsplit_once('/') {
            bases.insert(base.to_string());
        }
    }
    for base in bases {
        let (_, t0) = load_tensor_f32(&st, &format!("{base}/0"))?;
        let (_, t1) = load_tensor_f32(&st, &format!("{base}/1"))?;
        let (_, t2) = load_tensor_f32(&st, &format!("{base}/2"))?;
        let rem: Vec<f32> = t0
            .iter()
            .zip(t1.iter())
            .zip(t2.iter())
            .map(|((&a, &b), &c)| a + b + c)
            .collect();
        map.insert(base, rem);
    }
    Ok(map)
}

fn export_dit(
    writer: &mut StreamingHimgWriter,
    prefix: &str,
    dit_st: &SafeTensors<'_>,
    norm_rem: &HashMap<String, Vec<f32>>,
    lora: Option<(&SafeTensors<'_>, f32, usize)>,
    dtype: TensorDType,
) -> Result<()> {
    let get_with_rem = |key: &str| -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, mut vals) = load_tensor_f32(dit_st, key)?;
        if let Some(rem) = norm_rem.get(key) {
            for (v, &r) in vals.iter_mut().zip(rem.iter()) {
                *v += r;
            }
        }
        Ok((shape, vals))
    };

    // Keep small input/output/time/modulation projections in F32 for full precision
    for (dst, src) in [
        ("img_in", "dit.img_in.weight"),
        ("txt_in.text_norm", "dit.txt_in.text_norm.weight"),
        ("txt_in.in_layer", "dit.txt_in.in_layer.weight"),
        ("txt_in.out_layer", "dit.txt_in.out_layer.weight"),
        ("time_lin1", "dit.time_text_embed.timestep_embedder.linear_1.weight"),
        ("time_lin2", "dit.time_text_embed.timestep_embedder.linear_2.weight"),
        ("modulation", "dit.modulation.1.weight"),
        ("norm_out_lin", "dit.norm_out.linear.weight"),
        ("proj_out", "dit.proj_out.weight"),
    ] {
        let (shape, vals) = get_with_rem(src)?;
        writer.add_f32(&format!("{prefix}.{dst}"), shape, &vals)?;
    }

    let get_linear_maybe_lora = |base_key: &str| -> Result<(usize, usize, Vec<f32>)> {
        let (shape, mut w) = load_tensor_f32(dit_st, base_key)?;
        let (n, k) = (shape[0], shape[1]);
        if let Some((lora_st, scale, rank)) = lora {
            let stem = base_key.strip_suffix(".weight").unwrap();
            let a_key = format!("base_model.model.{stem}.lora_A.weight");
            let b_key = format!("base_model.model.{stem}.lora_B.weight");
            if lora_st.tensor(&a_key).is_ok() {
                let (_, a) = load_tensor_f32(lora_st, &a_key)?;
                let (_, b) = load_tensor_f32(lora_st, &b_key)?;
                merge_lora_inplace(&mut w, &b, &a, n, k, rank, scale);
            }
        }
        Ok((n, k, w))
    };

    for l in 0..DIT_LAYERS {
        let bp = format!("dit.transformer_blocks.{l}");
        let op = format!("{prefix}.blocks.{l}");

        let (sq, wq) = get_with_rem(&format!("{bp}.attn.norm_q.weight"))?;
        writer.add_f32(&format!("{op}.norm_q"), sq, &wq)?;
        let (sk, wk) = get_with_rem(&format!("{bp}.attn.norm_k.weight"))?;
        writer.add_f32(&format!("{op}.norm_k"), sk, &wk)?;

        // Fuse Q, K, V -> [3456, 1152]
        let (nq, kq, q) = get_linear_maybe_lora(&format!("{bp}.attn.to_q.weight"))?;
        let (nk, _, k) = get_linear_maybe_lora(&format!("{bp}.attn.to_k.weight"))?;
        let (nv, _, v) = get_linear_maybe_lora(&format!("{bp}.attn.to_v.weight"))?;
        let mut qkv = Vec::with_capacity((nq + nk + nv) * kq);
        qkv.extend_from_slice(&q);
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        write_linear(writer, &format!("{op}.qkv"), &qkv, nq + nk + nv, kq, dtype)?;

        let (no, ko, wo) = get_linear_maybe_lora(&format!("{bp}.attn.to_out.0.weight"))?;
        write_linear(writer, &format!("{op}.to_out"), &wo, no, ko, dtype)?;

        // Fuse Gate, Proj -> [6912, 1152]
        let (ng, kg, wg) = get_linear_maybe_lora(&format!("{bp}.img_mlp.gate_layer.weight"))?;
        let (np, _, wp) = get_linear_maybe_lora(&format!("{bp}.img_mlp.proj.weight"))?;
        let mut gu = Vec::with_capacity((ng + np) * kg);
        gu.extend_from_slice(&wg);
        gu.extend_from_slice(&wp);
        write_linear(writer, &format!("{op}.gate_up"), &gu, ng + np, kg, dtype)?;

        let (nd, kd, wd) = get_linear_maybe_lora(&format!("{bp}.img_mlp.out.weight"))?;
        write_linear(writer, &format!("{op}.mlp_out"), &wd, nd, kd, dtype)?;
    }

    Ok(())
}

fn export_qwen3(
    writer: &mut StreamingHimgWriter,
    st: &SafeTensors<'_>,
    dtype: TensorDType,
    quant_mode: QuantMode,
) -> Result<()> {
    let (emb_shape, emb) = load_tensor_f32(st, "model.embed_tokens.weight")?;
    if matches!(quant_mode, QuantMode::F32) {
        writer.add_f32("qwen3.embed_tokens", emb_shape, &emb)?;
    } else {
        let packed = pack_emb_q8(&emb, emb_shape[0], emb_shape[1]);
        writer.add_raw("qwen3.embed_tokens", TensorDType::EmbQ8, emb_shape, &packed)?;
    }

    for l in 0..QWEN3_LAYERS {
        let lp = format!("model.layers.{l}");
        let op = format!("qwen3.layers.{l}");

        let (s, v) = load_tensor_f32(st, &format!("{lp}.input_layernorm.weight"))?;
        writer.add_f32(&format!("{op}.input_layernorm"), s, &v)?;
        let (s, v) = load_tensor_f32(st, &format!("{lp}.post_attention_layernorm.weight"))?;
        writer.add_f32(&format!("{op}.post_attention_layernorm"), s, &v)?;
        let (s, v) = load_tensor_f32(st, &format!("{lp}.self_attn.q_norm.weight"))?;
        writer.add_f32(&format!("{op}.q_norm"), s, &v)?;
        let (s, v) = load_tensor_f32(st, &format!("{lp}.self_attn.k_norm.weight"))?;
        writer.add_f32(&format!("{op}.k_norm"), s, &v)?;

        // Fuse Q, K, V -> [4096, 1024]
        let (sq, q) = load_tensor_f32(st, &format!("{lp}.self_attn.q_proj.weight"))?;
        let (sk, k) = load_tensor_f32(st, &format!("{lp}.self_attn.k_proj.weight"))?;
        let (sv, v) = load_tensor_f32(st, &format!("{lp}.self_attn.v_proj.weight"))?;
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(&q);
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        write_linear(
            writer,
            &format!("{op}.qkv"),
            &qkv,
            sq[0] + sk[0] + sv[0],
            sq[1],
            dtype,
        )?;

        let (so, o) = load_tensor_f32(st, &format!("{lp}.self_attn.o_proj.weight"))?;
        write_linear(writer, &format!("{op}.o_proj"), &o, so[0], so[1], dtype)?;

        // Fuse gate_proj + up_proj -> [6144, 1024]
        let (sg, g) = load_tensor_f32(st, &format!("{lp}.mlp.gate_proj.weight"))?;
        let (su, u) = load_tensor_f32(st, &format!("{lp}.mlp.up_proj.weight"))?;
        let mut gu = Vec::with_capacity(g.len() + u.len());
        gu.extend_from_slice(&g);
        gu.extend_from_slice(&u);
        write_linear(
            writer,
            &format!("{op}.gate_up"),
            &gu,
            sg[0] + su[0],
            sg[1],
            dtype,
        )?;

        let (sd, d) = load_tensor_f32(st, &format!("{lp}.mlp.down_proj.weight"))?;
        write_linear(writer, &format!("{op}.down_proj"), &d, sd[0], sd[1], dtype)?;
    }

    let (sn, vn) = load_tensor_f32(st, "model.norm.weight")?;
    writer.add_f32("qwen3.norm", sn, &vn)?;
    Ok(())
}

fn export_vae_decoder(
    writer: &mut StreamingHimgWriter,
    st: &SafeTensors<'_>,
    vae_config_path: &Path,
    dtype: TensorDType,
) -> Result<()> {
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vae_config_path)?)?;
    writer.set_meta("vae_latents_mean", cfg_json["latents_mean"].clone());
    writer.set_meta("vae_latents_std", cfg_json["latents_std"].clone());

    let mut names: Vec<String> = st.names().into_iter().cloned().collect();
    names.sort();

    for name in names {
        // Strip encoder, quant_conv, and unused video time_conv layers!
        if name.starts_with("encoder.")
            || name.starts_with("quant_conv.")
            || name.contains("time_conv")
        {
            continue;
        }
        let dst_name = format!("vae.{name}");
        let (shape, vals) = load_tensor_f32(st, &name)?;
        if name.ends_with(".weight") && shape.len() == 4 {
            let (c_out, c_in, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
            let mat = if kh == 1 && kw == 1 {
                vals
            } else {
                transpose_conv_oihw_to_ohwi(&vals, c_out, c_in, kh, kw)
            };
            let k_dim = c_in * kh * kw;
            // Keep sensitive boundary convolutions (post_quant_conv, conv_in, conv_out) in F32!
            let keep_f32 = name == "post_quant_conv.weight"
                || name == "decoder.conv_in.weight"
                || name == "decoder.conv_out.weight";
            let use_dt = if keep_f32 { TensorDType::F32 } else { dtype };
            write_linear(writer, &dst_name, &mat, c_out, k_dim, use_dt)?;
        } else {
            // Biases and norm gammas -> flatten to 1D F32
            let flat_len = vals.len();
            writer.add_f32(&dst_name, vec![flat_len], &vals)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    println!(
        "Converting Haidass-Image-Instruct + Qwen3-0.6B with quant={:?}...",
        args.quant
    );

    let dit_file = File::open(args.model_dir.join("dit.safetensors"))?;
    let dit_mmap = unsafe { Mmap::map(&dit_file)? };
    let dit_st = SafeTensors::deserialize(&dit_mmap)?;

    let norm_rem = load_norm_remainders(&args.model_dir.join("dit_norm_precision.safetensors"))?;
    println!("Loaded {} FP32 norm remainder tensors.", norm_rem.len());

    let qwen_file = File::open(args.text_encoder_dir.join("model.safetensors"))?;
    let qwen_mmap = unsafe { Mmap::map(&qwen_file)? };
    let qwen_st = SafeTensors::deserialize(&qwen_mmap)?;

    let vae_file = File::open(args.model_dir.join("vae/diffusion_pytorch_model.safetensors"))?;
    let vae_mmap = unsafe { Mmap::map(&vae_file)? };
    let vae_st = SafeTensors::deserialize(&vae_mmap)?;

    let mut writer = StreamingHimgWriter::create(&args.output, args.quant.as_str())?;

    // Embed tokenizer.json inside metadata so the .himg file is 100% self-contained!
    let tok_json_str = std::fs::read_to_string(args.text_encoder_dir.join("tokenizer.json"))?;
    writer.set_meta(
        "tokenizer_json",
        serde_json::Value::String(tok_json_str),
    );

    let t_qwen = Instant::now();
    export_qwen3(&mut writer, &qwen_st, args.quant.qwen3_dtype(), args.quant)?;
    println!("  [1/3] Qwen3-0.6B packed in {:.2}s", t_qwen.elapsed().as_secs_f64());

    let t_dit = Instant::now();
    export_dit(&mut writer, "dit", &dit_st, &norm_rem, None, args.quant.dit_dtype())?;
    println!("  [2/3] DiT Base (34L) packed in {:.2}s", t_dit.elapsed().as_secs_f64());

    let t_vae = Instant::now();
    export_vae_decoder(
        &mut writer,
        &vae_st,
        &args.model_dir.join("vae/config.json"),
        args.quant.vae_dtype(),
    )?;
    println!("  [3/3] VAE Decoder packed in {:.2}s", t_vae.elapsed().as_secs_f64());

    writer.finish()?;
    let main_mb = std::fs::metadata(&args.output)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "Wrote main bundle: {} ({:.1} MB) in {:.2}s",
        args.output.display(),
        main_mb,
        t0.elapsed().as_secs_f64()
    );

    if let Some(writing_out) = args.writing_output {
        let lora_path = args.model_dir.join("writing_lora/adapter_model.safetensors");
        let lora_cfg_path = args.model_dir.join("writing_lora/adapter_config.json");
        if lora_path.is_file() {
            let tw = Instant::now();
            let cfg_json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&lora_cfg_path)?)?;
            let rank = cfg_json["r"].as_u64().unwrap_or(128) as usize;
            let alpha = cfg_json["lora_alpha"].as_f64().unwrap_or(256.0) as f32;
            let scale = alpha / (rank as f32);

            let lora_file = File::open(&lora_path)?;
            let lora_mmap = unsafe { Mmap::map(&lora_file)? };
            let lora_st = SafeTensors::deserialize(&lora_mmap)?;

            let mut w_writer = StreamingHimgWriter::create(&writing_out, args.quant.as_str())?;
            export_dit(
                &mut w_writer,
                "dit_writing",
                &dit_st,
                &norm_rem,
                Some((&lora_st, scale, rank)),
                args.quant.dit_dtype(),
            )?;
            w_writer.finish()?;
            let w_mb = std::fs::metadata(&writing_out)?.len() as f64 / (1024.0 * 1024.0);
            println!(
                "Wrote merged writing-LoRA bundle: {} ({:.1} MB, rank={}, alpha={}) in {:.2}s",
                writing_out.display(),
                w_mb,
                rank,
                alpha,
                tw.elapsed().as_secs_f64()
            );
        } else {
            println!("Skipping writing LoRA (not found at {})", lora_path.display());
        }
    }

    Ok(())
}
