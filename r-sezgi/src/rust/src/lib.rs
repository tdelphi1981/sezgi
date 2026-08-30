use savvy::savvy;

mod anytime;
mod bias;
mod experiment;
mod mo;
mod problems;
mod session;
mod solve;
mod stats;

/// Version of the underlying sezgi Rust core.
///
/// @returns A character scalar with the crate version string.
/// @export
#[savvy]
fn sezgi_version() -> savvy::Result<savvy::Sexp> {
    let v = env!("CARGO_PKG_VERSION").to_string();
    v.try_into()
}
