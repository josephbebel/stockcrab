// Stockcrab, a Rust port of Stockfish, a UCI chess playing engine derived from Glaurung 2.1
// Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
//
// Stockcrab is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Stockcrab is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Captures reproducible build metadata for `misc::engine_version_info`.
//!
//! Emits (via `cargo:rustc-env`):
//! * `STOCKCRAB_GIT_HASH`: short commit hash of the source tree, or `nogit`
//!   when git or the repository is unavailable (e.g. a crates.io tarball).
//! * `STOCKCRAB_UPSTREAM_REF`: the upstream Stockfish commit this port is
//!   synced to, read from `config/upstream.env` alongside the default network
//!   and bench expectation.
//!
//! * `STOCKCRAB_RUSTC_VERSION`: `rustc --version` of the compiler in use, for
//!   the `compiler` command (upstream prints the compiler name and version).
//!
//! Emits (via `cargo:rustc-cfg`):
//! * `stockcrab_dotprod`: on AArch64, when the target already has the `dotprod`
//!   feature (Apple Silicon by default, or `-C target-cpu`/`target-feature`) or
//!   the `neon-dotprod` Cargo feature asserts it. Selects the SDOT NNUE kernels,
//!   like `dotprod=yes` in the upstream Makefile.
//!
//! No wall-clock date is embedded, so rebuilding the same source gives the
//! same binary identity.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn upstream_setting(text: &str, name: &str) -> String {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value.to_owned())
        .unwrap_or_else(|| panic!("missing {name} in config/upstream.env"))
}

fn rustc_version() -> String {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.contains('\n'))
        .unwrap_or_else(|| "rustc".to_string())
}

/// Whether the AArch64 dot-product (SDOT) kernels can be compiled in: the
/// target guarantees `dotprod`, or the user asserted it with `neon-dotprod`.
fn aarch64_dotprod() -> bool {
    let var = |name: &str| std::env::var(name).unwrap_or_default();
    var("CARGO_CFG_TARGET_ARCH") == "aarch64"
        && (std::env::var_os("CARGO_FEATURE_NEON_DOTPROD").is_some()
            || var("CARGO_CFG_TARGET_FEATURE")
                .split(',')
                .any(|feature| feature == "dotprod"))
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(stockcrab_dotprod)");
    if aarch64_dotprod() {
        println!("cargo:rustc-cfg=stockcrab_dotprod");
    }
    println!(
        "cargo:rustc-env=STOCKCRAB_RUSTC_VERSION={}",
        rustc_version()
    );
    println!("cargo:rerun-if-changed=config/upstream.env");
    let upstream = std::fs::read_to_string("config/upstream.env").expect("upstream configuration");
    let revision = upstream_setting(&upstream, "STOCKFISH_REVISION");
    let network = upstream_setting(&upstream, "STOCKCRAB_NETWORK");
    let bench = upstream_setting(&upstream, "BENCH_NODES");
    assert_eq!(revision.len(), 40, "pin the complete upstream commit");
    assert!(bench.parse::<u64>().is_ok_and(|nodes| nodes > 0));
    println!("cargo:rustc-env=STOCKCRAB_DEFAULT_NETWORK={network}");
    println!("cargo:rustc-env=STOCKCRAB_BENCH_NODES={bench}");

    let hash = git(&["rev-parse", "--short=8", "HEAD"]).unwrap_or_else(|| "nogit".to_string());
    // Re-run when the checked-out commit changes: HEAD itself (checkout), the
    // branch ref it points to, and the HEAD reflog (updated by every commit,
    // which also covers refs kept in packed-refs).
    let mut watch = vec![git(&["rev-parse", "--git-path", "HEAD"])];
    if let Some(refname) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watch.push(git(&["rev-parse", "--git-path", &refname]));
    }
    watch.push(git(&["rev-parse", "--git-path", "logs/HEAD"]));
    for path in watch.into_iter().flatten() {
        if Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(path) = git(&["rev-parse", "--git-path", "packed-refs"])
        && Path::new(&path).exists()
    {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rustc-env=STOCKCRAB_GIT_HASH={hash}");
    println!("cargo:rustc-env=STOCKCRAB_UPSTREAM_REF={revision}");
}
