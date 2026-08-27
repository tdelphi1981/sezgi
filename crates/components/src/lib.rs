pub mod boundary;
pub mod init;
pub mod replace;
pub mod step;
pub mod de;
pub mod presets;

use sezgi_core::component::Registry;

pub fn register_builtins(reg: &mut Registry) {
    init::register(reg);
    boundary::register(reg);
    replace::register(reg);
    step::register(reg);
    de::register(reg);
}
