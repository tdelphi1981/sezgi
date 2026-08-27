pub mod boundary;
pub mod init;
pub mod replace;
pub mod step;

use sezgi_core::component::Registry;

pub fn register_builtins(reg: &mut Registry) {
    init::register(reg);
    boundary::register(reg);
    replace::register(reg);
    step::register(reg);
}
