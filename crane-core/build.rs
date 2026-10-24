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

/// Non-recursive `.cu` file listing for a single directory, sorted for
/// deterministic build output.
#[cfg(feature = "cuda")]
fn cu_files_in(dir: &str) -> Vec<std::path::PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    paths.sort();
    paths
}

/// Compiles `kernels/cuda/`'s (and `kernels/cuda/fattn/`'s) `.cu` sources
/// into the `crane_kernels_ptx.rs` bindings `ops::fused_ops`'s CUDA
/// dispatchers `include!`. `flash_attention/` holds a separately built host
/// launch adapter (cudaforge + CUTLASS, gated behind
/// `flash-attn`/`flash-inference`) that must not go through this PTX-only
/// pipeline, so it's deliberately not globbed here.
#[cfg(feature = "cuda")]
fn build_cuda_ptx() {
    use std::env;
    use std::path::PathBuf;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut kernel_paths = cu_files_in("kernels/cuda");
    kernel_paths.extend(cu_files_in("kernels/cuda/fattn"));

    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernel_paths)
        .arg("--expt-relaxed-constexpr")
        .arg("--extended-lambda")
        .arg("-std=c++17")
        .arg("-O3");

    let bindings = builder.build_ptx().expect("Failed to compile CUDA kernels");
    bindings
        .write(out_dir.join("crane_kernels_ptx.rs"))
        .expect("Failed to write PTX bindings");
}

fn main() {
    #[cfg(any(feature = "flash-attn", feature = "flash-inference"))]
    {
        use std::path::PathBuf;
        let headers = std::env::var_os("CRANE_FLASH_ATTN_HEADERS").map(PathBuf::from).unwrap_or_else(|| {
            let cargo_cache = std::env::var_os("CARGO_HOME").map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").expect("HOME or CARGO_HOME required")).join(".cargo")
            });
            std::fs::read_dir(cargo_cache.join("registry/src")).expect("Cargo registry source directory")
                .filter_map(|entry|entry.ok().map(|e|e.path().join("candle-flash-attn-0.11.0/kernels")))
                .find(|path|path.join("flash_fwd_launch_template.h").is_file())
                .expect("unmodified candle-flash-attn 0.11.0 headers; or set CRANE_FLASH_ATTN_HEADERS")
        });
        println!("cargo::rerun-if-env-changed=CRANE_FLASH_ATTN_HEADERS");
        println!("cargo::rerun-if-changed=kernels/cuda/flash_attention/");
        let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
        cudaforge::KernelBuilder::new()
            .source_files(["kernels/cuda/flash_attention/launch.cu"])
            .out_dir(&out)
            .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
            .arg(&format!("-I{}", headers.display()))
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--fmad=true")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .build_lib(out.join("libcrane_flash_split.a"))
            .expect("build Crane split-KV launcher");
        println!("cargo::rustc-link-search=native={}", out.display());
        println!("cargo::rustc-link-lib=static=crane_flash_split");
        println!("cargo::rustc-link-lib=dylib=cudart");
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }

    println!("cargo::rerun-if-changed=kernels/cuda/");
    println!("cargo::rerun-if-changed=kernels/sycl/");
    // The Metal kernels in `kernels/metal/fused_ops.metal` are `include_str!`'d
    // at compile time and JIT-compiled at first dispatch through
    // `MetalDevice::new_library_with_source(...)`. This directive here only
    // re-runs `build.rs` if the file changes — `include_str!` itself invalidates
    // the cargo build cache when the file is modified, but listing the
    // directory explicitly also covers any future sibling .metal files that
    // might be added without being referenced.
    println!("cargo::rerun-if-changed=kernels/metal/");
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "onnx")]
    {
        println!("cargo::rerun-if-changed=src/onnx/onnx.proto3");
        prost_build::compile_protos(&["src/onnx/onnx.proto3"], &["src/onnx"])
            .expect("failed to generate Crane's vendored ONNX protobuf bindings");
    }

    // Only compile CUDA kernels when the cuda feature is enabled.
    #[cfg(feature = "cuda")]
    build_cuda_ptx();

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
        let sources = [
            "kernels/sycl/gdn.cpp",
            "kernels/sycl/fused_ops.cpp",
            "kernels/sycl/quant_iq.cpp",
            "kernels/sycl/hyper_connection.cpp",
            "kernels/sycl/qsa_mask.cpp",
            "kernels/sycl/topk.cpp",
            "kernels/sycl/dspark.cpp",
        ];
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
