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

//! Meta-tests: check that the test suite actually runs.
//!
//! Two kinds of mistake leave a green `cargo test` run meaningless without any
//! visible sign:
//!
//! * code gated behind `#[cfg(feature = "...")]` naming a feature that
//!   Cargo.toml does not declare, which is permanently compiled out;
//! * `debug_assert!` inside test code, which expands to nothing in release
//!   builds (the mode this project normally tests in).
//!
//! These tests scan the source tree for both patterns, plus unexplained
//! `#[ignore]`s, and catch the mistake when it is introduced.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in ["src", "tests"] {
        collect(&crate_root().join(dir), &mut out);
    }
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Crudely strip line comments and string literals so the scanners below match
/// real code rather than prose about the patterns they look for.
fn code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_str = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                chars.next(); // skip the escaped character
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '/' if chars.peek() == Some(&'/') => break, // line comment
            _ => out.push(c),
        }
    }
    out
}

/// Strip line comments but keep string literals, for scanners that need the
/// contents of a string (such as a cfg feature name).
fn strip_comments(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_str = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_str {
            out.push(c);
            if c == '\\' {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

/// Read a source file, failing loudly so an unreadable file cannot make a
/// scanner pass vacuously.
fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn rel(p: &Path) -> String {
    p.strip_prefix(crate_root())
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Features declared in Cargo.toml's `[features]` table.
fn declared_features() -> BTreeSet<String> {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).expect("Cargo.toml");
    let mut out = BTreeSet::new();
    let mut in_features = false;

    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if !in_features || t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = t.split_once('=') {
            out.insert(name.trim().trim_matches('"').to_string());
        }
    }
    out
}

/// Every `#[cfg(feature = "...")]` (or `cfg!(feature = "...")`) must name a feature that exists.
///
/// A typo or a never-declared feature does not fail the build: the gate simply
/// evaluates to false forever and the code inside disappears.
#[test]
fn every_cfg_feature_is_declared_in_cargo_toml() {
    let declared = declared_features();
    let mut problems = Vec::new();

    for path in rust_sources() {
        let text = read_source(&path);
        for (i, raw) in text.lines().enumerate() {
            // Comments only: the feature name lives inside a string literal.
            let line = strip_comments(raw);
            // Only genuine cfg attributes and `cfg!` expressions;
            // `#![feature(...)]` is the unrelated unstable-language-feature
            // namespace.
            if !(line.contains("#[cfg") || line.contains("#![cfg") || line.contains("cfg!(")) {
                continue;
            }

            let mut rest = line.as_str();
            while let Some(idx) = rest.find("feature = \"") {
                // `target_feature = "avx2"` is a CPU capability, not a cargo feature.
                let is_target = rest[..idx].ends_with("target_");
                rest = &rest[idx + "feature = \"".len()..];
                let Some(end) = rest.find('"') else { break };
                let name = &rest[..end];
                rest = &rest[end + 1..];

                if is_target || declared.contains(name) {
                    continue;
                }
                problems.push(format!(
                    "{}:{}: cfg(feature = \"{name}\") is not declared in Cargo.toml",
                    rel(&path),
                    i + 1
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "code gated behind undeclared cargo features can never compile:\n  {}\n\
         Declared features: {:?}",
        problems.join("\n  "),
        declared
    );
}

/// Test bodies must not use `debug_assert!`, which is a no-op in release.
///
/// The project's documented workflow is `cargo test --release`, so a
/// `debug_assert` in a test asserts nothing there. This scans `tests/` (every
/// item is a test) and `#[cfg(test)]` modules in `src/`. Test-only items outside
/// a test module (a `#[cfg(test)] fn`, say) are not scanned.
#[test]
fn no_debug_assert_inside_test_code() {
    let mut problems = Vec::new();

    for path in rust_sources() {
        let text = read_source(&path);
        let is_integration_test = rel(&path).starts_with("tests/");

        // Track whether we are inside a `#[cfg(test)]` module by brace depth.
        let mut test_mod_depth: Option<i32> = None;
        let mut depth: i32 = 0;
        let mut pending_cfg_test = false;

        for (i, line) in text.lines().enumerate() {
            let t = line.trim();

            // The item text after an optional leading `#[cfg(test)]`.
            let item = match t.strip_prefix("#[cfg(test)]") {
                Some(rest) => {
                    pending_cfg_test = true;
                    rest.trim()
                }
                None => t,
            };
            let is_mod_decl = item.starts_with("mod ")
                || item.starts_with("pub mod ")
                || item.starts_with("pub(crate) mod ");
            // The attribute applied to something other than a module (the brace
            // of a `mod` may sit on the next line, so a lone `{` keeps it armed).
            if pending_cfg_test
                && !item.is_empty()
                && !item.starts_with("#[")
                && !item.starts_with("//")
                && !item.starts_with('{')
                && !is_mod_decl
            {
                pending_cfg_test = false;
            }
            // `#[cfg(test)] mod tests;` has no body to track.
            if is_mod_decl && item.ends_with(';') {
                pending_cfg_test = false;
            }

            let in_test_code = is_integration_test || test_mod_depth.is_some_and(|d| depth >= d);

            let code = code_only(line);
            // Require the macro invocation, not merely the identifier (this
            // test's own name contains the word).
            let is_invocation = code.contains("debug_assert!")
                || code.contains("debug_assert_eq!")
                || code.contains("debug_assert_ne!");
            if in_test_code && is_invocation {
                problems.push(format!(
                    "{}:{}: debug_assert in test code is a no-op under `cargo test --release`",
                    rel(&path),
                    i + 1
                ));
            }

            let opens = code.matches('{').count() as i32;
            let closes = code.matches('}').count() as i32;

            if pending_cfg_test && opens > 0 && (is_mod_decl || item.starts_with('{')) {
                test_mod_depth = Some(depth + 1);
                pending_cfg_test = false;
            }

            depth += opens - closes;
            if let Some(d) = test_mod_depth
                && depth < d
            {
                test_mod_depth = None;
            }
        }
    }

    assert!(
        problems.is_empty(),
        "assertions that vanish in release builds:\n  {}",
        problems.join("\n  ")
    );
}

/// Every `#[ignore]` must carry a reason on the same line.
///
/// An unexplained `#[ignore]` is indistinguishable from an abandoned test.
#[test]
fn every_ignored_test_states_why() {
    let mut problems = Vec::new();

    for path in rust_sources() {
        let text = read_source(&path);
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if !t.starts_with("#[ignore") {
                continue;
            }
            // Accept either `#[ignore = "..."]` or a trailing `// reason`.
            let has_reason = t.contains('=') || t.contains("//");
            if !has_reason {
                problems.push(format!(
                    "{}:{}: #[ignore] with no stated reason",
                    rel(&path),
                    i + 1
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "unexplained #[ignore]s are indistinguishable from abandoned tests:\n  {}",
        problems.join("\n  ")
    );
}

/// Sanity check on the scanner itself: it must be finding real files.
///
/// Without this, a broken path or a silent `read_dir` failure would make every
/// test above pass by scanning nothing.
#[test]
fn scanner_sees_the_source_tree() {
    let files = rust_sources();
    assert!(
        files.len() > 20,
        "only found {} rust files; the scanner is not walking the tree",
        files.len()
    );
    assert!(
        files.iter().any(|p| rel(p) == "src/search.rs"),
        "src/search.rs not found by the scanner"
    );
    assert!(
        !declared_features().is_empty(),
        "no features parsed from Cargo.toml; the parser is broken"
    );
}
