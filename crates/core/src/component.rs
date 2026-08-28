use crate::problem::{Evaluator, Population};
use crate::rng::RngStream;
use crate::space::{Genotype, SearchSpace};
use crate::state::{Blackboard, StateReq};
use std::collections::HashMap;

pub struct Ctx<'a, 'b> {
    pub space: &'a SearchSpace,
    pub rng: &'a mut RngStream,
    pub bb: &'a mut Blackboard,
    pub eval: &'a mut Evaluator<'b>,
    pub iteration: u64,
}

#[derive(Debug, Clone)]
pub enum SupportedBlocks {
    All,
    Only(Vec<&'static str>),
}

#[derive(Debug, Clone)]
pub struct ComponentMeta {
    pub kind: &'static str,
    pub supported_blocks: SupportedBlocks,
    pub requires: Vec<StateReq>,
    pub provides: Vec<StateReq>,
}

impl ComponentMeta {
    pub fn supports_block(&self, tag: &str) -> bool {
        match &self.supported_blocks {
            SupportedBlocks::All => true,
            SupportedBlocks::Only(tags) => tags.contains(&tag),
        }
    }
}

pub trait Initializer: Send + Sync {
    fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype>;
    fn meta(&self) -> ComponentMeta;
}

pub trait Generator: Send + Sync {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype>;
    fn meta(&self) -> ComponentMeta;
}

pub trait Replacer: Send + Sync {
    fn replace(&self, pop: &mut Population, off_ind: Vec<Genotype>, off_fit: Vec<f64>, ctx: &mut Ctx);
    fn meta(&self) -> ComponentMeta;
}

pub trait BoundaryHandler: Send + Sync {
    fn repair(&self, g: &mut Genotype, space: &SearchSpace, ctx: &mut Ctx);
    fn meta(&self) -> ComponentMeta;
}

#[derive(Debug, thiserror::Error)]
pub enum ComponentError {
    #[error("unknown component kind: {0}")]
    UnknownKind(String),
    #[error("invalid parameter ({kind}): {reason}")]
    InvalidParams { kind: String, reason: String },
}

type Factory<T> = Box<dyn Fn(&serde_json::Value) -> Result<Box<T>, ComponentError> + Send + Sync>;

#[derive(Default)]
pub struct Registry {
    initializers: HashMap<String, Factory<dyn Initializer>>,
    generators: HashMap<String, Factory<dyn Generator>>,
    replacers: HashMap<String, Factory<dyn Replacer>>,
    boundaries: HashMap<String, Factory<dyn BoundaryHandler>>,
}

macro_rules! reg_family {
    ($reg_fn:ident, $build_fn:ident, $field:ident, $trait_:ident) => {
        pub fn $reg_fn<F>(&mut self, kind: &str, f: F)
        where
            F: Fn(&serde_json::Value) -> Result<Box<dyn $trait_>, ComponentError>
                + Send
                + Sync
                + 'static,
        {
            self.$field.insert(kind.to_string(), Box::new(f));
        }
        pub fn $build_fn(
            &self,
            kind: &str,
            params: &serde_json::Value,
        ) -> Result<Box<dyn $trait_>, ComponentError> {
            self.$field
                .get(kind)
                .ok_or_else(|| ComponentError::UnknownKind(kind.to_string()))?(params)
        }
    };
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }
    reg_family!(
        register_initializer,
        build_initializer,
        initializers,
        Initializer
    );
    reg_family!(
        register_generator,
        build_generator,
        generators,
        Generator
    );
    reg_family!(register_replacer, build_replacer, replacers, Replacer);
    reg_family!(
        register_boundary,
        build_boundary,
        boundaries,
        BoundaryHandler
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{BlockValues, Genotype};

    struct DummyInit;
    impl Initializer for DummyInit {
        fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype> {
            (0..n)
                .map(|_| Genotype {
                    blocks: vec![BlockValues::Float(vec![ctx.rng.next_f64()])],
                })
                .collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta {
                kind: "dummy-init",
                supported_blocks: SupportedBlocks::Only(vec!["float"]),
                requires: vec![],
                provides: vec![],
            }
        }
    }

    #[test]
    fn registry_builds_registered_kind() {
        let mut reg = Registry::new();
        reg.register_initializer("dummy-init", |_params| Ok(Box::new(DummyInit)));
        assert!(reg
            .build_initializer("dummy-init", &serde_json::json!({}))
            .is_ok());
    }

    #[test]
    fn unknown_kind_is_error() {
        let reg = Registry::new();
        let e = reg.build_initializer("yok-boyle-bilesen", &serde_json::json!({}));
        assert!(matches!(e, Err(ComponentError::UnknownKind(_))));
    }

    #[test]
    fn supported_blocks_check() {
        let m = DummyInit.meta();
        assert!(m.supports_block("float"));
        assert!(!m.supports_block("permutation"));
    }
}
