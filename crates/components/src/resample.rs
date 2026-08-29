use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::Genotype;
use crate::init::sample_uniform;

pub struct UniformResampleGenerator;

impl Generator for UniformResampleGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        (0..pop.len()).map(|_| sample_uniform(ctx.space, ctx.rng)).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/uniform-resample", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/uniform-resample", |_| Ok(Box::new(UniformResampleGenerator)));
}
