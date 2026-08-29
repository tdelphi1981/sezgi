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
    /// Smallest population size this component can operate on. Checked by
    /// `AlgorithmSpec::validate` (static, pre-run) as well as by each
    /// component's own runtime assertions (dynamic backstop, since some
    /// requirements — e.g. Nelder-Mead's `pop_size >= dim + 1` — depend on
    /// the search space and can't be expressed as a fixed constant here).
    pub min_pop: usize,
}

impl ComponentMeta {
    /// Start a meta with empty `requires`/`provides` and `min_pop: 1`. Chain
    /// `.with_requires(..)`, `.with_provides(..)`, `.with_min_pop(..)` to set
    /// only the fields that differ from those defaults.
    pub fn new(kind: &'static str, supported_blocks: SupportedBlocks) -> Self {
        Self {
            kind,
            supported_blocks,
            requires: vec![],
            provides: vec![],
            min_pop: 1,
        }
    }

    pub fn with_requires(mut self, requires: Vec<StateReq>) -> Self {
        self.requires = requires;
        self
    }

    pub fn with_provides(mut self, provides: Vec<StateReq>) -> Self {
        self.provides = provides;
        self
    }

    pub fn with_min_pop(mut self, min_pop: usize) -> Self {
        self.min_pop = min_pop;
        self
    }

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

pub trait Adapter: Send + Sync {
    /// Runs after the stage's replacer, once per iteration. May mutate the
    /// population (e.g. L-SHADE shrinking) and blackboard state.
    fn adapt(&self, pop: &mut Population, ctx: &mut Ctx);
    fn meta(&self) -> ComponentMeta;
}

/// What a restart component asks the engine to do.
///
/// `new_pop_size == 0` means "keep the current population size" — the engine
/// re-initializes at the population's current size rather than treating 0 as
/// a literal (and invalid) target size.
pub struct RestartDirective {
    pub new_pop_size: usize,
}

pub trait Restart: Send + Sync {
    /// Checked once per iteration, after all stages. Some(_) = restart now.
    fn check(&self, pop: &Population, ctx: &mut Ctx) -> Option<RestartDirective>;
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
    adapters: HashMap<String, Factory<dyn Adapter>>,
    restarts: HashMap<String, Factory<dyn Restart>>,
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
    reg_family!(register_adapter, build_adapter, adapters, Adapter);
    reg_family!(register_restart, build_restart, restarts, Restart);
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
            ComponentMeta::new("dummy-init", SupportedBlocks::Only(vec!["float"]))
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
