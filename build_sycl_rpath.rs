// Shared build-script helper, `include!`d by the binary crates.
//
// A dependency's `cargo::rustc-link-arg` does not reach the final binary's
// link, so the out-of-tree SYCL kernel libraries (`libcandle_sycl.so`,
// `libcrane_gdn_sycl.so`) built into other crates' `OUT_DIR`s would only be
// findable via `LD_LIBRARY_PATH`. `cargo::rustc-link-arg-bins` does apply to
// this package's binaries, so bake those directories in as an rpath and the
// binaries run straight from `target/release/`.
//
// The directories are siblings of this crate's own `OUT_DIR`
// (`<target>/<profile>/build/<pkg>-<hash>/out`), and the dependencies' build
// scripts have already run by the time this one does.
#[allow(dead_code)]
fn emit_sycl_rpath() {
    let Ok(out_dir) = std::env::var("OUT_DIR") else {
        return;
    };
    let out = std::path::PathBuf::from(&out_dir);
    // .../build/<pkg>-<hash>/out -> .../build
    let Some(build_root) = out.parent().and_then(std::path::Path::parent) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(build_root) else {
        return;
    };

    // Newest first, so a stale OUT_DIR from an earlier build never shadows the
    // library this build just produced.
    let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path().join("out");
        for lib in ["libcandle_sycl.so", "libcrane_gdn_sycl.so"] {
            let candidate = dir.join(lib);
            if let Ok(meta) = std::fs::metadata(&candidate) {
                let when = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                let d = dir.display().to_string();
                if !found.iter().any(|(_, p)| *p == d) {
                    found.push((when, d));
                }
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));

    // `--disable-new-dtags` emits the old DT_RPATH rather than DT_RUNPATH:
    // RUNPATH applies only to the object carrying it, while RPATH is searched
    // for dependencies further down the chain, which the `dlopen`ed SYCL
    // adapter and its own dependencies need.
    println!("cargo::rustc-link-arg-bins=-Wl,--disable-new-dtags");
    for (_, dir) in found {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{dir}");
        println!("cargo::rerun-if-changed={dir}");
    }
    for dir in oneapi_runtime_dirs() {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
}

/// oneAPI runtime library directories present on this machine.
#[allow(dead_code)]
fn oneapi_runtime_dirs() -> Vec<String> {
    let root = std::env::var("ONEAPI_ROOT").ok().or_else(|| {
        std::path::Path::new("/opt/intel/oneapi")
            .exists()
            .then(|| "/opt/intel/oneapi".to_string())
    });
    let Some(root) = root else {
        return Vec::new();
    };
    [
        "compiler/latest/lib",
        "mkl/latest/lib",
        "mkl/latest/lib/intel64",
        "tcm/latest/lib",
        "umf/latest/lib",
        "tbb/latest/lib/intel64/gcc4.8",
    ]
    .iter()
    .map(|sub| format!("{root}/{sub}"))
    .filter(|p| std::path::Path::new(p).exists())
    .collect()
}
