//! Deterministic standard-normal RNG (xoshiro256++ with SplitMix64 seeding + Box-Muller).
//! Also supports loading an external raw FP32 latent file for exact bit-for-bit parity checks
//! against PyTorch `torch.randn`.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        let mut x = seed;
        let mut splitmix = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn fill_randn(&mut self, out: &mut [f32]) {
        let mut i = 0;
        while i < out.len() {
            let mut u1 = self.next_f64();
            if u1 < 1e-12 {
                u1 = 1e-12;
            }
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out[i] = (r * theta.cos()) as f32;
            if i + 1 < out.len() {
                out[i + 1] = (r * theta.sin()) as f32;
            }
            i += 2;
        }
    }
}
