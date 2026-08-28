use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum SpaceError {
    #[error("invalid bounds: block {index}: lo={lo} >= hi={hi}")]
    InvalidBounds { index: usize, lo: f64, hi: f64 },
    #[error("invalid genotype: {reason}")]
    InvalidGenotype { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Float { lo: f64, hi: f64, n: usize },
    Int { lo: i64, hi: i64, n: usize },
    Categorical { k: u32, n: usize },
    Permutation { n: usize },
    Binary { n: usize },
}

impl Block {
    pub fn dim(&self) -> usize {
        match *self {
            Block::Float { n, .. } | Block::Int { n, .. }
            | Block::Categorical { n, .. } | Block::Permutation { n }
            | Block::Binary { n } => n,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchSpace { blocks: Vec<Block> }

#[derive(Debug, Clone, PartialEq)]
pub enum BlockValues {
    Float(Vec<f64>), Int(Vec<i64>), Cat(Vec<u32>), Perm(Vec<u32>), Bin(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Genotype { pub blocks: Vec<BlockValues> }

impl SearchSpace {
    pub fn new(blocks: Vec<Block>) -> Result<Self, SpaceError> {
        for (index, b) in blocks.iter().enumerate() {
            match *b {
                Block::Float { lo, hi, .. } if lo >= hi =>
                    return Err(SpaceError::InvalidBounds { index, lo, hi }),
                Block::Int { lo, hi, .. } if lo >= hi =>
                    return Err(SpaceError::InvalidBounds { index, lo: lo as f64, hi: hi as f64 }),
                _ => {}
            }
        }
        Ok(Self { blocks })
    }

    pub fn blocks(&self) -> &[Block] { &self.blocks }
    pub fn dim(&self) -> usize { self.blocks.iter().map(Block::dim).sum() }

    pub fn validate(&self, g: &Genotype) -> Result<(), SpaceError> {
        let err = |reason: String| Err(SpaceError::InvalidGenotype { reason });
        if g.blocks.len() != self.blocks.len() {
            return err(format!("block count {} != {}", g.blocks.len(), self.blocks.len()));
        }
        for (i, (b, v)) in self.blocks.iter().zip(&g.blocks).enumerate() {
            let ok = match (b, v) {
                (Block::Float { lo, hi, n }, BlockValues::Float(xs)) =>
                    xs.len() == *n && xs.iter().all(|x| x.is_finite() && *lo <= *x && x <= hi),
                (Block::Int { lo, hi, n }, BlockValues::Int(xs)) =>
                    xs.len() == *n && xs.iter().all(|x| lo <= x && x <= hi),
                (Block::Categorical { k, n }, BlockValues::Cat(xs)) =>
                    xs.len() == *n && xs.iter().all(|x| x < k),
                (Block::Permutation { n }, BlockValues::Perm(xs)) => {
                    let mut seen = vec![false; *n];
                    xs.len() == *n && xs.iter().all(|&x| {
                        let x = x as usize;
                        x < *n && !std::mem::replace(&mut seen[x], true)
                    })
                }
                (Block::Binary { n }, BlockValues::Bin(xs)) => xs.len() == *n,
                _ => false,
            };
            if !ok { return err(format!("block {i} type/value mismatch")); }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed_space() -> SearchSpace {
        SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: 3 },
            Block::Permutation { n: 4 },
        ]).unwrap()
    }

    #[test]
    fn dim_sums_blocks() {
        assert_eq!(mixed_space().dim(), 7);
    }

    #[test]
    fn invalid_bounds_rejected() {
        let e = SearchSpace::new(vec![Block::Float { lo: 1.0, hi: -1.0, n: 2 }]);
        assert!(matches!(e, Err(SpaceError::InvalidBounds { .. })));
    }

    #[test]
    fn validate_checks_permutation() {
        let s = mixed_space();
        let bad = Genotype { blocks: vec![
            BlockValues::Float(vec![0.0, 0.0, 0.0]),
            BlockValues::Perm(vec![0, 0, 2, 3]), // repeated 0: not a permutation
        ]};
        assert!(matches!(s.validate(&bad), Err(SpaceError::InvalidGenotype { .. })));
    }

    #[test]
    fn validate_checks_block_arity_and_type() {
        let s = mixed_space();
        let bad = Genotype { blocks: vec![BlockValues::Float(vec![0.0; 3])] };
        assert!(s.validate(&bad).is_err());
    }
}
