pub mod boundary;
pub mod init;
pub mod replace;
pub mod step;
pub mod de;
pub mod ga;
pub mod pso;
pub mod sa;
pub mod resample;
pub mod shade;
pub mod presets;
pub mod linalg;
pub mod cma;
pub mod restart;
pub mod nm;
pub mod gwo;
pub mod woa;
pub mod hs;
pub mod cs;
pub mod goa;
pub mod sca;
pub mod jaya;

use sezgi_core::component::Registry;

pub fn register_builtins(reg: &mut Registry) {
    init::register(reg);
    boundary::register(reg);
    replace::register(reg);
    step::register(reg);
    de::register(reg);
    ga::register(reg);
    pso::register(reg);
    sa::register(reg);
    resample::register(reg);
    shade::register(reg);
    cma::register(reg);
    restart::register(reg);
    nm::register(reg);
    gwo::register(reg);
    woa::register(reg);
    hs::register(reg);
    cs::register(reg);
    goa::register(reg);
    sca::register(reg);
    jaya::register(reg);
}
