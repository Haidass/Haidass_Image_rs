//! Ultra-fast Rust CPU Inference Engine for DALabCommunity/Haidass-Image-Instruct.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use haidass_image_rs::dit::{DitModel, SampleConfig, Sampler, DIT_IN_CH};
use haidass_image_rs::format::HimgFile;
use haidass_image_rs::qwen3::{Qwen3Encoder, QWEN3_EOS_ID, QWEN3_MAX_LEN};
use haidass_image_rs::rng::Rng;
use haidass_image_rs::routing::route_prompt;
use haidass_image_rs::vae::VaeDecoder;
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SamplerArg {
    Euler,
    Midpoint,
    Heun,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Preset {
    /// Exact reference settings: Euler 50 steps, CFG 8.0, no caching
    Quality,
    /// Balanced fast mode: Euler 28 steps, CFG 8.0, light TeaCache (0.08) + late CFG skip (<=0.10)
    Balanced,
    /// Ultra-fast turbo mode: Euler 20 steps, CFG 7.5, TeaCache (0.14) + late CFG skip (<=0.18)
    Fast,
}

#[derive(Parser, Debug)]
#[command(name = "haidass-infer", about = "Pure Rust AVX-VNNI CPU inference engine for Haidass-Image-Instruct")]
struct Args {
    /// Text prompt (use quotes inside prompt to auto-activate writing LoRA + spelling)
    prompt: String,

    /// Path to main quantized .himg bundle
    #[arg(short, long)]
    model: PathBuf,

    /// Optional path to pre-merged writing-LoRA .himg (auto-detected if sibling *-writing*.himg exists)
    #[arg(long)]
    writing_model: Option<PathBuf>,

    /// Output PNG file path
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,

    /// Random seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Optional preset overriding steps/cache/cfg-skip unless explicitly set
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Denoising steps (default: 50)
    #[arg(long)]
    steps: Option<usize>,

    /// Classifier-free guidance scale (default: 8.0)
    #[arg(long)]
    cfg: Option<f32>,

    /// Timestep shift (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    shift: f32,

    /// ODE sampler
    #[arg(long, value_enum, default_value_t = SamplerArg::Euler)]
    sampler: SamplerArg,

    /// TeaCache relative-L1 skip threshold (0.0 = disabled/exact; try 0.08..0.15 for 1.5-2x speedup)
    #[arg(long)]
    cache_thresh: Option<f32>,

    /// Skip CFG (unconditional pass) when t <= value (0.0 = disabled; try 0.12..0.20)
    #[arg(long)]
    cfg_skip_below: Option<f32>,

    /// Output image resolution (square, must be multiple of 16; model trained at 512)
    #[arg(long, default_value_t = 512)]
    resolution: usize,

    /// Number of Rayon CPU threads (default: auto)
    #[arg(long)]
    threads: Option<usize>,

    /// Optional raw f32 latent input file (64*32*32 f32s) for exact parity testing against PyTorch
    #[arg(long)]
    init_latent: Option<PathBuf>,

    /// Optional path to dump final latent as raw f32 for diagnostics
    #[arg(long)]
    dump_latent: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .context("Failed to configure Rayon thread pool")?;
    }

    let t_total = Instant::now();
    println!(
        "Haidass-Image-RS | threads={} | avxvnni={} | avx2={}",
        rayon::current_num_threads(),
        cfg_avxvnni(),
        cfg_avx2()
    );

    // 1. Prompt routing & spelling annotation
    let (processed_prompt, use_writing) = route_prompt(&args.prompt);
    println!("Prompt (writing_lora={}): {}", use_writing, processed_prompt);

    // 2. Open mmap'd .himg bundle
    let t_load = Instant::now();
    let main_file = HimgFile::open(&args.model)?;
    println!(
        "Opened {} (quant={}) in {:.1} ms",
        args.model.display(),
        main_file.header.quant_mode,
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    // 3. Tokenize prompt using embedded Qwen3 tokenizer.json
    let tok_json = main_file
        .header
        .metadata
        .get("tokenizer_json")
        .and_then(|v| v.as_str())
        .context("Missing embedded tokenizer_json in .himg")?;
    let tokenizer = Tokenizer::from_bytes(tok_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Tokenizer load error: {e}"))?;

    let enc = tokenizer
        .encode(processed_prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;
    let mut cond_ids: Vec<u32> = enc.get_ids().to_vec();
    if cond_ids.len() > QWEN3_MAX_LEN {
        cond_ids.truncate(QWEN3_MAX_LEN);
    }
    if cond_ids.is_empty() {
        cond_ids.push(QWEN3_EOS_ID);
    }
    let uncond_ids = [QWEN3_EOS_ID];
    println!("Tokenized prompt: {} tokens", cond_ids.len());

    // 4. Qwen3-0.6B Text Encoding
    let t_enc = Instant::now();
    let encoder = Qwen3Encoder::load(&main_file)?;
    let cond_ctx = encoder.encode(&cond_ids)?;
    let uncond_ctx = encoder.encode(&uncond_ids)?;
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    println!("Text encoding (Qwen3-0.6B): {:.1} ms", enc_ms);

    // 5. Select DiT model (base vs writing-LoRA merged)
    let writing_file_holder: Option<HimgFile>;
    let dit = if use_writing {
        let w_path = args.writing_model.clone().or_else(|| {
            let stem = args.model.file_stem()?.to_str()?;
            let cand = args.model.with_file_name(format!("{stem}-writing.himg"));
            cand.is_file().then_some(cand)
        });
        if let Some(wp) = w_path {
            println!("Loading merged writing-LoRA backbone: {}", wp.display());
            writing_file_holder = Some(HimgFile::open(&wp)?);
            DitModel::load(writing_file_holder.as_ref().unwrap(), "dit_writing")?
        } else {
            println!("Warning: quoted text detected but no writing-LoRA .himg found; falling back to base DiT.");
            writing_file_holder = None;
            DitModel::load(&main_file, "dit")?
        }
    } else {
        writing_file_holder = None;
        DitModel::load(&main_file, "dit")?
    };

    // 6. Initialize latent noise [64, H_lat, W_lat]
    let lat_h = args.resolution / 16;
    let lat_w = args.resolution / 16;
    let mut latent = vec![0.0f32; DIT_IN_CH * lat_h * lat_w];
    if let Some(lat_path) = &args.init_latent {
        let bytes = std::fs::read(lat_path)?;
        let f: &[f32] = bytemuck::cast_slice(&bytes);
        anyhow::ensure!(
            f.len() == latent.len(),
            "init_latent size {} != expected {}",
            f.len(),
            latent.len()
        );
        latent.copy_from_slice(f);
    } else {
        let mut rng = Rng::from_seed(args.seed);
        rng.fill_randn(&mut latent);
    }

    // 7. Build sampling config (applying preset + explicit overrides)
    let mut scfg = SampleConfig {
        latent_h: lat_h,
        latent_w: lat_w,
        timestep_shift: args.shift,
        sampler: match args.sampler {
            SamplerArg::Euler => Sampler::Euler,
            SamplerArg::Midpoint => Sampler::Midpoint,
            SamplerArg::Heun => Sampler::Heun,
        },
        ..Default::default()
    };
    if let Some(preset) = args.preset {
        match preset {
            Preset::Quality => {
                scfg.steps = 50;
                scfg.cfg = 8.0;
                scfg.cache_thresh = 0.0;
                scfg.cfg_skip_below = 0.0;
            }
            Preset::Balanced => {
                scfg.steps = 28;
                scfg.cfg = 8.0;
                scfg.cache_thresh = 0.08;
                scfg.cfg_skip_below = 0.10;
            }
            Preset::Fast => {
                scfg.steps = 20;
                scfg.cfg = 7.5;
                scfg.cache_thresh = 0.14;
                scfg.cfg_skip_below = 0.18;
            }
        }
    }
    if let Some(s) = args.steps {
        scfg.steps = s;
    }
    if let Some(c) = args.cfg {
        scfg.cfg = c;
    }
    if let Some(ct) = args.cache_thresh {
        scfg.cache_thresh = ct;
    }
    if let Some(cs) = args.cfg_skip_below {
        scfg.cfg_skip_below = cs;
    }

    println!(
        "Sampling: {}x{} | steps={} | cfg={} | sampler={:?} | cache_thresh={} | cfg_skip_below={}",
        args.resolution,
        args.resolution,
        scfg.steps,
        scfg.cfg,
        scfg.sampler,
        scfg.cache_thresh,
        scfg.cfg_skip_below
    );

    let stats = dit.sample(
        &mut latent,
        &cond_ctx,
        cond_ids.len(),
        &uncond_ctx,
        uncond_ids.len(),
        &scfg,
        |step, total, ms| {
            if step == 1 || step % 5 == 0 || step == total {
                println!("  Step {:2}/{} ({:.0} ms)", step, total, ms);
            }
        },
    );
    println!(
        "DiT done: prefill={:.1} ms | denoise={:.2} s | evals={} | cached={} | cfg_skipped={} | avg={:.0} ms/eval",
        stats.prefill_ms,
        stats.denoise_ms / 1000.0,
        stats.total_evals,
        stats.cached_steps,
        stats.cfg_skipped_steps,
        stats.denoise_ms / (stats.total_evals.max(1) as f64)
    );

    if let Some(dump_path) = &args.dump_latent {
        std::fs::write(dump_path, bytemuck::cast_slice(&latent))?;
    }

    // 8. VAE Decode -> PNG
    let t_vae = Instant::now();
    let vae = VaeDecoder::load(&main_file)?;
    let (rgb, img_h, img_w) = vae.decode_to_rgb(&latent, lat_h, lat_w);
    let vae_ms = t_vae.elapsed().as_secs_f64() * 1000.0;
    println!("VAE decode ({}x{}): {:.2} s", img_w, img_h, vae_ms / 1000.0);

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(img_w as u32, img_h as u32, rgb)
            .context("Failed to construct RGB image buffer")?;
    img.save(&args.output)?;

    println!(
        "Saved {} | TOTAL end-to-end: {:.2} s",
        args.output.display(),
        t_total.elapsed().as_secs_f64()
    );

    drop(writing_file_holder);
    Ok(())
}

fn cfg_avxvnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn cfg_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
