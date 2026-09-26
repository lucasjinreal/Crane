include!("../build_sycl_rpath.rs");

fn main() {
    // Build scripts are compiled without the crate's features, so `cfg(feature
    // = "sycl")` is always false here; Cargo passes enabled features as env
    // vars instead.
    if std::env::var_os("CARGO_FEATURE_SYCL").is_some() {
        emit_sycl_rpath();
    }
}
