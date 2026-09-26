//! Making a `--features sycl` binary runnable without sourcing `setvars.sh`.

/// Set up the SYCL runtime environment for the current process: put the
/// oneAPI runtime directories on `LD_LIBRARY_PATH` (re-executing once if they
/// are missing) and apply the Level-Zero adapter workaround below. Together
/// these make a `--features sycl` binary runnable straight from
/// `target/release/` with nothing sourced first.
///
/// An rpath on the binary is not enough for `LD_LIBRARY_PATH`: the SYCL
/// runtime `dlopen`s a Unified Runtime adapter whose `libumf.so.1` carries
/// `DT_RUNPATH [$ORIGIN]`, and an object with RUNPATH stops the loader
/// consulting the executable's RPATH for that object's dependencies — so
/// `libhwloc.so.15` never resolves, the adapter fails to load, and SYCL
/// silently reports no devices. The loader reads `LD_LIBRARY_PATH` at process
/// start, so a single idempotent re-exec is the self-contained fix; an
/// explicit `LD_LIBRARY_PATH` from the caller is left alone.
///
/// Call it first thing in `main`, before anything touches SYCL.
pub fn ensure_sycl_runtime_env() {
    #[cfg(all(feature = "sycl", target_os = "linux"))]
    {
        use std::os::unix::process::CommandExt;

        // oneAPI 2026.x defaults to the Level-Zero V2 adapter, which fails on
        // the first kernel submission from an in-order USM queue on
        // Battlemage (Arc Pro B70) with UR_RESULT_ERROR_UNSUPPORTED_FEATURE
        // (44) — reproduces with a bare 10-line SYCL program, so it's a
        // driver/adapter limitation, not a Crane bug. The UR loader reads
        // this lazily (after main starts, not at process launch like
        // LD_LIBRARY_PATH above), so a plain `set_var` is enough — no
        // re-exec needed; an explicit override from the caller is left
        // alone.
        if std::env::var_os("UR_LOADER_USE_LEVEL_ZERO_V2").is_none() {
            // SAFETY: called first thing in `main`, before any other thread
            // exists to race this write.
            unsafe { std::env::set_var("UR_LOADER_USE_LEVEL_ZERO_V2", "0") };
        }

        const MARKER: &str = "CRANE_SYCL_ENV_APPLIED";
        if std::env::var_os(MARKER).is_some() {
            return;
        }
        let dirs: Vec<&str> = option_env!("CRANE_SYCL_RUNTIME_DIRS")
            .unwrap_or_default()
            .split(':')
            .filter(|d| !d.is_empty())
            .collect();
        if dirs.is_empty() {
            return;
        }
        let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if dirs.iter().all(|d| current.split(':').any(|p| p == *d)) {
            return; // already set up, e.g. the caller sourced setvars.sh
        }

        let mut combined = dirs.join(":");
        if !current.is_empty() {
            combined.push(':');
            combined.push_str(&current);
        }
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let err = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env("LD_LIBRARY_PATH", combined)
            .env(MARKER, "1")
            .exec();
        eprintln!(
            "crane: could not re-exec with the oneAPI runtime path ({err}); SYCL may fall \
             back to CPU. Workaround: source /opt/intel/oneapi/setvars.sh, then add the \
             libcandle_sycl.so / libcrane_gdn_sycl.so build directories to LD_LIBRARY_PATH \
             before running."
        );
    }
}
