#!/bin/bash
# Type-check every target configuration this crate supports. SIMD paths are
# selected at compile time, so a change that builds on one configuration can
# break another (this has happened: an import used only by the aarch64 NEON
# module). Host-aware: on an x86-64 host the x86 rows use the host target; on any other
# host (e.g. Apple Silicon) they are replaced by host rows plus cross-checked x86-64 SIMD
# rows (baseline, v2, v3, AVX-512 with VNNI, Ice Lake, AVX-512F without BW) for
# x86_64-unknown-linux-gnu and/or x86_64-apple-darwin. Those cross rows need the target to be
# installed; a missing one prints `skipped` and does not fail the script, so read the output:
# a run without them does NOT cover the x86 SIMD paths. A row is `ok` only when it compiles
# without warnings (`WARNINGS` and `FAILED` both make the script exit 1).
# Needs: rustup target add aarch64-unknown-linux-gnu aarch64-apple-darwin \
#   x86_64-pc-windows-gnu, and on a non-x86 host also x86_64-unknown-linux-gnu (or
#   x86_64-apple-darwin).
# Runs under bash 3.2 (macOS /bin/bash).
#
# Usage: scripts/check-all.sh [extra cargo args]
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
logdir=$(mktemp -d) && trap 'rm -rf "$logdir"' EXIT
run() { # name target-dir rustflags [--target T]
  local name=$1 dir=$2 flags=$3; shift 3
  printf '%-28s' "$name"
  if RUSTFLAGS="$flags" CARGO_TARGET_DIR="target/check-$dir" \
       cargo check --all-targets "$@" ${EXTRA:-} >"$logdir/$dir.log" 2>&1; then
    # A warning is a failure too: it would otherwise be invisible (the log is discarded).
    if grep -q '^warning' "$logdir/$dir.log"; then
      echo WARNINGS; grep -E '^warning' -A6 "$logdir/$dir.log" | head -30; fail=1
    else
      echo ok
    fi
  else
    echo FAILED; grep -E '^error' -A5 "$logdir/$dir.log" | head -20; fail=1
  fi
}
EXTRA="$*"
host_arch=$(uname -m)
case "$host_arch" in
  x86_64|amd64) x86_host=1 ;;
  *)            x86_host=0 ;;
esac
# Rows that need an x86-64 host (host-target builds with x86 target-cpu flags).
if [ "$x86_host" = 1 ]; then
  run "x86-64 baseline"  base   ""
  EXTRA="--release $*" run "x86-64 release"   rel    ""
  EXTRA="--release $*" run "x86-64 native rel." relnat "-C target-cpu=native"
  run "x86-64-v2"        v2     "-C target-cpu=x86-64-v2"
  run "x86-64-v3"        v3     "-C target-cpu=x86-64-v3"
  run "AVX-512F without BW" f-no-bw "-C target-feature=+avx512f,-avx512bw"
  run "x86-64 native"    native "-C target-cpu=native"
else
  # Non-x86 host (e.g. aarch64): the host-target rows are the native ones, and
  # x86-only -C target-cpu values do not exist.
  run "host ($host_arch)"         host    ""
  EXTRA="--release $*" run "host ($host_arch) rel."  hostrel ""
  run "host native"      hostnat "-C target-cpu=native"
  # x86-64 SIMD paths, cross-checked (`cargo check` needs no linker). With --target, RUSTFLAGS
  # reach the target crates only, so build scripts still run with host code.
  installed=$(rustup target list --installed 2>/dev/null || true)
  has_target() { printf '%s\n' "$installed" | grep -qx "$1"; }
  x86_cross=""
  for t in x86_64-unknown-linux-gnu x86_64-apple-darwin; do
    if has_target "$t"; then
      if [ -z "$x86_cross" ]; then
        # Full SIMD matrix on the first installed target.
        x86_cross=$t
        run "x86-64 baseline (cross)" xbase "" --target "$t"
        run "x86-64-v2 (cross)"   xv2   "-C target-cpu=x86-64-v2" --target "$t"
        run "x86-64-v3 (cross)"   xv3   "-C target-cpu=x86-64-v3" --target "$t"
        run "AVX-512 VNNI (cross)" xv4vnni \
          "-C target-cpu=x86-64-v4 -C target-feature=+avx512vnni" --target "$t"
        run "AVX-512 ICL (cross)" xicl  "-C target-cpu=icelake-server" --target "$t"
        run "AVX-512F no BW (cross)" xfnobw \
          "-C target-feature=+avx512f,-avx512bw" --target "$t"
      else
        # The other OS: one AVX2 row for its cfg(target_os) paths.
        run "x86-64-v3 macOS (cross)" xv3mac "-C target-cpu=x86-64-v3" --target "$t"
      fi
    fi
  done
  if [ -z "$x86_cross" ]; then
    printf '%-28s%s\n' "x86-64 SIMD (cross)" \
      "skipped (rustup target add x86_64-unknown-linux-gnu): x86 SIMD paths NOT checked"
  fi
fi
run "aarch64 linux"    a64    "" --target aarch64-unknown-linux-gnu
run "aarch64 linux dotprod" a64dp "" --target aarch64-unknown-linux-gnu --features neon-dotprod
# The Apple target has dotprod by default (SDOT kernels); keep a plain-NEON row.
run "aarch64 macOS plain" a64mac "-C target-feature=-dotprod" --target aarch64-apple-darwin
run "aarch64 macOS dotprod" a64macdp "" --target aarch64-apple-darwin --features neon-dotprod
run "x86-64 windows"   win    "" --target x86_64-pc-windows-gnu
exit $fail
