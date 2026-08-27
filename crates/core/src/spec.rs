use crate::component::{ComponentError, ComponentMeta, Registry};
use crate::space::{Block, SearchSpace};
use serde::{Deserialize, Serialize};

fn block_tag(b: &Block) -> &'static str {
    match b {
        Block::Float { .. } => "float",
        Block::Int { .. } => "int",
        Block::Categorical { .. } => "categorical",
        Block::Permutation { .. } => "permutation",
        Block::Binary { .. } => "binary",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentSpec {
    pub kind: String,
    #[serde(flatten, default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageSpec {
    pub generator: ComponentSpec,
    pub replacer: ComponentSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TerminationSpec {
    pub budget: u64,
    #[serde(default)]
    pub target: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlgorithmSpec {
    pub name: String,
    pub pop_size: usize,
    pub init: ComponentSpec,
    pub boundary: ComponentSpec,
    pub stages: Vec<StageSpec>,
    pub termination: TerminationSpec,
}

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("spec ayrıştırma hatası: {0}")]
    Parse(String),
    #[error(transparent)]
    Component(#[from] ComponentError),
    #[error("bileşen `{kind}` `{block}` blok tipini desteklemiyor")]
    UnsupportedBlock { kind: String, block: String },
    #[error("bileşen `{kind}` `{key}` durum anahtarını istiyor ama hiçbir bileşen sağlamıyor")]
    MissingState { kind: String, key: String },
    #[error("durum anahtarı `{key}` tip uyumsuz: beklenen {expected}, bulunan {found}")]
    StateTypeMismatch { key: String, expected: String, found: String },
}

impl AlgorithmSpec {
    pub fn from_toml(src: &str) -> Result<Self, SpecError> {
        toml::from_str(src).map_err(|e| SpecError::Parse(e.to_string()))
    }
    pub fn from_json(src: &str) -> Result<Self, SpecError> {
        serde_json::from_str(src).map_err(|e| SpecError::Parse(e.to_string()))
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("spec serileştirilemedi")
    }

    pub fn validate(&self, reg: &Registry, space: &SearchSpace) -> Result<(), SpecError> {
        // Tüm bileşenleri kur, metaları topla
        let mut metas: Vec<ComponentMeta> = vec![
            reg.build_initializer(&self.init.kind, &self.init.params)?.meta(),
            reg.build_boundary(&self.boundary.kind, &self.boundary.params)?.meta(),
        ];
        for st in &self.stages {
            metas.push(reg.build_generator(&st.generator.kind, &st.generator.params)?.meta());
            metas.push(reg.build_replacer(&st.replacer.kind, &st.replacer.params)?.meta());
        }
        // (2) blok desteği
        for m in &metas {
            for b in space.blocks() {
                let tag = block_tag(b);
                if !m.supports_block(tag) {
                    return Err(SpecError::UnsupportedBlock {
                        kind: m.kind.to_string(), block: tag.to_string(),
                    });
                }
            }
        }
        // (3) durum beyanları
        let provides: Vec<_> = metas.iter().flat_map(|m| m.provides.clone()).collect();
        for m in &metas {
            for req in &m.requires {
                match provides.iter().find(|p| p.key == req.key) {
                    None => return Err(SpecError::MissingState {
                        kind: m.kind.to_string(), key: req.key.clone(),
                    }),
                    Some(p) if p.type_name != req.type_name =>
                        return Err(SpecError::StateTypeMismatch {
                            key: req.key.clone(),
                            expected: req.type_name.to_string(),
                            found: p.type_name.to_string(),
                        }),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::*;
    use crate::space::{Block, Genotype, SearchSpace};
    use crate::state::StateReq;

    // Test bileşenleri: perm desteklemeyen jeneratör + durum beyanlı çift
    struct FloatOnlyGen;
    impl Generator for FloatOnlyGen {
        fn generate(&self, _p: &crate::problem::Population, _c: &mut Ctx) -> Vec<Genotype> { vec![] }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "float-only",
                supported_blocks: SupportedBlocks::Only(vec!["float"]),
                requires: vec![StateReq::of::<Vec<f64>>("velocity")], provides: vec![] }
        }
    }
    struct NoopReplacer;
    impl Replacer for NoopReplacer {
        fn replace(&self, _p: &mut crate::problem::Population, _i: Vec<Genotype>,
                   _f: Vec<f64>, _c: &mut Ctx) {}
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "noop", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }
    struct VelInit;
    impl Initializer for VelInit {
        fn initialize(&self, _n: usize, _c: &mut Ctx) -> Vec<Genotype> { vec![] }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "vel-init", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![StateReq::of::<Vec<f64>>("velocity")] }
        }
    }
    struct NoopBoundary;
    impl BoundaryHandler for NoopBoundary {
        fn repair(&self, _g: &mut Genotype, _s: &SearchSpace, _c: &mut Ctx) {}
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "noop-b", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.register_generator("float-only", |_| Ok(Box::new(FloatOnlyGen)));
        r.register_replacer("noop", |_| Ok(Box::new(NoopReplacer)));
        r.register_initializer("vel-init", |_| Ok(Box::new(VelInit)));
        r.register_boundary("noop-b", |_| Ok(Box::new(NoopBoundary)));
        r
    }

    fn spec(init: &str) -> AlgorithmSpec {
        AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: init.into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "noop-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "float-only".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "noop".into(), params: serde_json::json!({}) },
            }],
            termination: TerminationSpec { budget: 100, target: None },
        }
    }

    #[test]
    fn toml_roundtrip() {
        let toml_src = r#"
            name = "l-shade-benzeri"
            pop_size = 20
            init = { kind = "vel-init" }
            boundary = { kind = "noop-b" }
            [[stages]]
            generator = { kind = "float-only", f = 0.5 }
            replacer = { kind = "noop" }
            [termination]
            budget = 1000
        "#;
        let s = AlgorithmSpec::from_toml(toml_src).unwrap();
        assert_eq!(s.stages[0].generator.params["f"], serde_json::json!(0.5));
        let s2 = AlgorithmSpec::from_json(&s.to_json()).unwrap();
        assert_eq!(s2.name, "l-shade-benzeri");
    }

    #[test]
    fn valid_spec_passes() {
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 2 }]).unwrap();
        assert!(spec("vel-init").validate(&registry(), &space).is_ok());
    }

    #[test]
    fn unsupported_block_rejected() {
        let space = SearchSpace::new(vec![Block::Permutation { n: 5 }]).unwrap();
        let e = spec("vel-init").validate(&registry(), &space);
        assert!(matches!(e, Err(SpecError::UnsupportedBlock { .. })));
    }

    #[test]
    fn missing_state_rejected() {
        // init "vel-init" yerine velocity SAĞLAMAYAN bir bileşen kullanılırsa
        // float-only'nin requires'ı karşılanamaz
        let mut r = registry();
        struct PlainInit;
        impl Initializer for PlainInit {
            fn initialize(&self, _n: usize, _c: &mut Ctx) -> Vec<Genotype> { vec![] }
            fn meta(&self) -> ComponentMeta {
                ComponentMeta { kind: "plain", supported_blocks: SupportedBlocks::All,
                    requires: vec![], provides: vec![] }
            }
        }
        r.register_initializer("plain", |_| Ok(Box::new(PlainInit)));
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 2 }]).unwrap();
        let e = spec("plain").validate(&r, &space);
        assert!(matches!(e, Err(SpecError::MissingState { .. })));
    }

    #[test]
    fn unknown_kind_rejected() {
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 2 }]).unwrap();
        let mut s = spec("vel-init");
        s.stages[0].generator.kind = "yok".into();
        assert!(matches!(s.validate(&registry(), &space), Err(SpecError::Component(_))));
    }
}
