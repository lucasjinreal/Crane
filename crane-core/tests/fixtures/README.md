# crane-core test fixtures

## `fattn_goldens_rocm.txt`, `fattn_goldens_cuda.txt`

Bit-exact goldens for the MMA flash-attention kernels, checked by
`kernel_output_matches_goldens` in
`src/ops/fused_ops/fattn_golden_tests.rs`. Each line holds the FNV-1a 64
hash of one case's output bits. Any kernel change that alters a single
output bit fails the test, without storing megabytes of reference output.

### Format

```text
# arch: cc 12.0
# Output hashes of the MMA flash-attention entries. ...
bf16 hd128 seq1024 seed0 bb4c4d1c3a96682f
bf16 hd128 seq2048 seed0 offset 540893bd0cbed8ce
```

- `# arch:` is the compute capability of the GPU the file was recorded on.
  The check refuses to run against a different architecture, because each
  architecture compiles the kernels to different code and may give
  different bits. HIP reports all gfx110x parts as 11.0, so it can't tell
  those apart.
- Each other line is `<dtype> hd<head_dim> seq<len> seed<n> [variant]
  <hash>`. Without a variant, the case is the plain causal entry over the
  full long-sequence sweep (`SEQS`, `DISTS`, `SEEDS_PER_DIST` in
  `fattn_long_seq_tests.rs`). The variants run at two lengths, one seed
  per distribution:
  - `offset`: causal over a cached prefix (`kv_offset > 0`, q shorter
    than k/v)
  - `windowed`: sliding window
  - `full`: no mask
  - `gqa2`: causal at GQA ratio 2 instead of 4

Each backend has one file, valid for one architecture. The committed
`fattn_goldens_rocm.txt` comes from an RDNA4 GPU (`cc 12.0`), and
`fattn_goldens_cuda.txt` from an NVIDIA L4 (`cc 8.9`). On any other GPU,
record a local copy to compare kernel changes against, and don't commit
it.

### Checking

```sh
# ROCm
cargo test -p crane-core --no-default-features --features "rocm,onnx" \
    --release --lib golden -- --ignored --nocapture
# CUDA
cargo test -p crane-core --features "cuda,flash-attn-intree" \
    --release --lib golden -- --ignored --nocapture
```

### Recording

Re-record only when a kernel change is meant to alter output bits. Set
`CRANE_FATTN_GOLDEN=record` through `--config`, since `cargo test` does
not forward inline environment variables:

```sh
cargo test --config 'env.CRANE_FATTN_GOLDEN="record"' \
    -p crane-core --no-default-features --features "rocm,onnx" \
    --release --lib golden -- --ignored --nocapture
```

Before overwriting, the run prints the old and new architecture and how
many hashes differ. Check that this matches the scope of the change. Any
other value of `CRANE_FATTN_GOLDEN` fails the test.

CI has no GPU, so nothing there runs these checks. Record the CUDA file on
real NVIDIA hardware, for example a Hugging Face Job on an `l4x1` flavor.
