use std::any::Any;
use std::collections::HashMap;

#[derive(Default)]
pub struct Blackboard {
    slots: HashMap<String, Box<dyn Any + Send>>,
}

/// Bileşenlerin durum beyanı (Task 6'daki Component trait'i kullanır).
#[derive(Debug, Clone, PartialEq)]
pub struct StateReq {
    pub key: String,
    pub type_name: &'static str,
}

impl StateReq {
    pub fn of<T: Any>(key: &str) -> Self {
        Self { key: key.to_string(), type_name: std::any::type_name::<T>() }
    }
}

impl Blackboard {
    pub fn new() -> Self { Self::default() }

    pub fn insert<T: Any + Send>(&mut self, key: &str, value: T) {
        self.slots.insert(key.to_string(), Box::new(value));
    }

    pub fn get<T: Any>(&self, key: &str) -> Option<&T> {
        self.slots.get(key)?.downcast_ref::<T>()
    }

    pub fn get_mut<T: Any>(&mut self, key: &str) -> Option<&mut T> {
        self.slots.get_mut(key)?.downcast_mut::<T>()
    }

    pub fn contains(&self, key: &str) -> bool { self.slots.contains_key(key) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_roundtrip() {
        let mut bb = Blackboard::new();
        bb.insert("velocity", vec![1.0f64, 2.0]);
        assert_eq!(bb.get::<Vec<f64>>("velocity").unwrap(), &vec![1.0, 2.0]);
    }

    #[test]
    fn wrong_type_returns_none() {
        let mut bb = Blackboard::new();
        bb.insert("velocity", vec![1.0f64]);
        assert!(bb.get::<Vec<i64>>("velocity").is_none());
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut bb = Blackboard::new();
        bb.insert("count", 0u64);
        *bb.get_mut::<u64>("count").unwrap() += 5;
        assert_eq!(*bb.get::<u64>("count").unwrap(), 5);
    }

    #[test]
    fn missing_key_is_none() {
        let bb = Blackboard::new();
        assert!(bb.get::<u64>("yok").is_none());
    }
}
