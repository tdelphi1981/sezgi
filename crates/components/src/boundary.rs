use sezgi_core::component::*;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

pub struct Clamp;

impl BoundaryHandler for Clamp {
    fn repair(&self, g: &mut Genotype, space: &SearchSpace, _ctx: &mut Ctx) {
        for (b, v) in space.blocks().iter().zip(&mut g.blocks) {
            match (b, v) {
                (Block::Float { lo, hi, .. }, BlockValues::Float(xs)) =>
                    xs.iter_mut().for_each(|x| *x = x.clamp(*lo, *hi)),
                (Block::Int { lo, hi, .. }, BlockValues::Int(xs)) =>
                    xs.iter_mut().for_each(|x| *x = (*x).clamp(*lo, *hi)),
                _ => {} // cat/perm/bin: yapısal olarak sınır dışına çıkamaz
            }
        }
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "boundary/clamp", supported_blocks: SupportedBlocks::All,
            requires: vec![], provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_boundary("boundary/clamp", |_| Ok(Box::new(Clamp)));
}
