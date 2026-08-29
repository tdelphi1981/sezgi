use sezgi_core::component::*;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

pub fn sample_uniform(space: &SearchSpace, rng: &mut RngStream) -> Genotype {
    Genotype {
        blocks: space.blocks().iter().map(|b| match *b {
            Block::Float { lo, hi, n } => BlockValues::Float(
                (0..n).map(|_| lo + (hi - lo) * rng.next_f64()).collect()),
            Block::Int { lo, hi, n } => BlockValues::Int(
                (0..n).map(|_| lo + rng.next_below((hi - lo + 1) as u64) as i64).collect()),
            Block::Categorical { k, n } => BlockValues::Cat(
                (0..n).map(|_| rng.next_below(k as u64) as u32).collect()),
            Block::Permutation { n } => {
                let mut xs: Vec<u32> = (0..n as u32).collect();
                for i in (1..n).rev() {                      // Fisher–Yates
                    let j = rng.next_below(i as u64 + 1) as usize;
                    xs.swap(i, j);
                }
                BlockValues::Perm(xs)
            }
            Block::Binary { n } => BlockValues::Bin(
                (0..n).map(|_| rng.next_f64() < 0.5).collect()),
        }).collect(),
    }
}

pub struct UniformInit;

impl Initializer for UniformInit {
    fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype> {
        (0..n).map(|_| sample_uniform(ctx.space, ctx.rng)).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("init/uniform", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_initializer("init/uniform", |_| Ok(Box::new(UniformInit)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::component::{Ctx, Initializer};
    use sezgi_core::problem::{Evaluator, SphereShifted};
    use sezgi_core::rng::RngStream;
    use sezgi_core::space::{Block, SearchSpace};
    use sezgi_core::state::Blackboard;

    #[test]
    fn uniform_init_valid_for_mixed_space() {
        let space = SearchSpace::new(vec![
            Block::Float { lo: -2.0, hi: 3.0, n: 2 },
            Block::Int { lo: 0, hi: 9, n: 2 },
            Block::Categorical { k: 4, n: 1 },
            Block::Permutation { n: 5 },
            Block::Binary { n: 3 },
        ]).unwrap();
        let p = SphereShifted::new(vec![0.5], -5.0, 5.0); // eval is unused, just fills out Ctx
        let mut eval = Evaluator::new(&p, 10);
        let mut rng = RngStream::from_master(1, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space: &space, rng: &mut rng, bb: &mut bb,
                            eval: &mut eval, iteration: 0 };
        let pop = UniformInit.initialize(50, &mut ctx);
        assert_eq!(pop.len(), 50);
        for g in &pop { space.validate(g).unwrap(); }
    }
}
