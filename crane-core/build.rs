/// oneAPI runtime library directories, baked into the built `.so` as an rpath
/// so the caller does not have to have sourced `setvars.sh`.
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
    ]
    .iter()
    .map(|sub| format!("{root}/{sub}"))
    .filter(|p| std::path::Path::new(p).exists())
    .collect()
}

fn main() {
    println!("cargo::rerun-if-changed=kernels/cuda/");
    println!("cargo::rerun-if-changed=kernels/sycl/");
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "onnx")]
    {
        println!("cargo::rerun-if-changed=src/onnx/onnx.proto3");
        prost_build::compile_protos(&["src/onnx/onnx.proto3"], &["src/onnx"])
            .expect("failed to generate Crane's vendored ONNX protobuf bindings");
    }

    // Only compile CUDA kernels when the cuda feature is enabled.
    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let builder = bindgen_cuda::Builder::default()
            .kernel_paths_glob("kernels/cuda/**/*.cu")
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3");

        let bindings = builder.build_ptx().expect("Failed to compile CUDA kernels");
        bindings
            .write(out_dir.join("crane_kernels_ptx.rs"))
            .expect("Failed to write PTX bindings");
    }

    // Compile the fused SYCL kernels into `libcrane_gdn_sycl.so` with `icpx`.
    // Only when `--features sycl` — a plain build never needs oneAPI. The
    // candle fork's `candle-sycl-kernels` is already built the same way, so
    // `icpx` is guaranteed to be on `PATH` (or `CANDLE_SYCL_ICPX`) here.
    #[cfg(feature = "sycl")]
    {
        use std::path::{Path, PathBuf};
        use std::process::Command;

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // All out-of-tree SYCL kernels build into one shared object.
        let sources = ["kernels/sycl/gdn.cpp", "kernels/sycl/fused_ops.cpp"];
        let lib = out_dir.join("libcrane_gdn_sycl.so");

        let icpx = std::env::var("CANDLE_SYCL_ICPX").unwrap_or_else(|_| {
            let root =
                std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| "/opt/intel/oneapi".to_string());
            let cand = format!("{root}/compiler/latest/bin/icpx");
            if Path::new(&cand).exists() {
                cand
            } else {
                "icpx".to_string()
            }
        });

        let status = Command::new(&icpx)
            .args([
                "-fsycl",
                "-O2",
                "-fPIC",
                "-std=c++20",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-Wno-unknown-pragmas",
                "-shared",
            ])
            .args(sources)
            .arg("-o")
            .arg(&lib)
            .args(
                oneapi_runtime_dirs()
                    .iter()
                    .map(|d| format!("-Wl,-rpath,{d}")),
            )
            .arg("-lsycl")
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn `{icpx}` for {sources:?}: {e}"));
        assert!(status.success(), "icpx failed to build {sources:?}");

        println!("cargo::rustc-link-search=native={}", out_dir.display());
        println!("cargo::rustc-link-lib=dylib=crane_gdn_sycl");
        // `rustc-link-arg` does not reach a *dependent* binary's link, so this
        // only covers `cargo test -p crane-core`. Binary crates bake the same
        // directory in themselves via `build_sycl_rpath.rs`.
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
        // Baked in so `ensure_sycl_runtime_env` re-execs with exactly the
        // directories this build linked against.
        println!(
            "cargo::rustc-env=CRANE_SYCL_RUNTIME_DIRS={}",
            oneapi_runtime_dirs().join(":")
        );
    }
}
