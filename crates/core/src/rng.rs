/// SplitMix64: seed mixing + stream derivation (Steele et al. 2014).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// xoshiro256++ (Blackman & Vigna 2019) — hand-written implementation, fixed.
#[derive(Debug, Clone)]
pub struct RngStream { s: [u64; 4], master: u64, path_digest: u64 }

impl RngStream {
    pub fn from_master(master: u64, path: &[u64]) -> Self {
        // fold the path down to a single digest via SplitMix64
        let mut digest = master;
        for &p in path {
            let mut st = digest ^ p.wrapping_mul(0x9E3779B97F4A7C15);
            digest = splitmix64(&mut st);
        }
        let mut st = digest;
        let s = [splitmix64(&mut st), splitmix64(&mut st),
                 splitmix64(&mut st), splitmix64(&mut st)];
        Self { s, master, path_digest: digest }
    }

    pub fn split(&self, child_id: u64) -> Self {
        let mut st = self.path_digest ^ child_id.wrapping_mul(0x9E3779B97F4A7C15);
        let digest = splitmix64(&mut st);
        let mut st2 = digest;
        let s = [splitmix64(&mut st2), splitmix64(&mut st2),
                 splitmix64(&mut st2), splitmix64(&mut st2)];
        Self { s, master: self.master, path_digest: digest }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = s[3].rotate_left(45);
        result
    }

    /// [0,1) — 53-bit precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// [0, n) integer — rejection sampling (no modulo bias).
    pub fn next_below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone { return v % n; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_path() {
        let a: Vec<u64> = { let mut r = RngStream::from_master(42, &[0, 1]); (0..4).map(|_| r.next_u64()).collect() };
        let b: Vec<u64> = { let mut r = RngStream::from_master(42, &[0, 1]); (0..4).map(|_| r.next_u64()).collect() };
        assert_eq!(a, b);
    }

    #[test]
    fn different_paths_diverge() {
        let mut a = RngStream::from_master(42, &[0, 1]);
        let mut b = RngStream::from_master(42, &[0, 2]);
        assert_ne!((0..4).map(|_| a.next_u64()).collect::<Vec<_>>(),
                   (0..4).map(|_| b.next_u64()).collect::<Vec<_>>());
    }

    #[test]
    fn split_equals_extended_path() {
        let mut a = RngStream::from_master(42, &[7]).split(3);
        let mut b = RngStream::from_master(42, &[7, 3]);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = RngStream::from_master(1, &[0]);
        for _ in 0..1000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    // Golden value: this test breaks if the implementation changes accidentally (spec §6).
    #[test]
    fn golden_values_pinned() {
        let mut r = RngStream::from_master(123, &[1, 2]);
        let got: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        assert_eq!(got, &[6760034611921890008, 14428432178303842706, 17525307277462374819]);
    }
}
