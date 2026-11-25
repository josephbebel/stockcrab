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

//! Helpers shared by the integration tests.

/// Called when the default NNUE network is not on disk.
///
/// By default the calling test prints a notice and is skipped, so `cargo test`
/// works on a fresh checkout. With `STOCKCRAB_REQUIRE_NET` set (CI does this)
/// a missing network is a failure instead, so the NNUE tests can never be
/// silently skipped. Fetch the network with `scripts/fetch-net.sh`.
// Not every test binary that includes this shared module calls this helper.
#[allow(dead_code)]
pub fn network_missing(test: &str) {
    if std::env::var_os("STOCKCRAB_REQUIRE_NET").is_some()
        || std::env::var("STOCKCRAB_REQUIRE_SYZYGY").is_ok_and(|value| value == "1")
    {
        panic!(
            "{test}: the default NNUE network is required but missing (run scripts/fetch-net.sh)"
        );
    }
    eprintln!("SKIP {test}: default NNUE network not found (run scripts/fetch-net.sh)");
}

/// Called when the pinned C++ Stockfish source tree is not available.
///
/// By default the calling test prints a notice and is skipped, so `cargo test`
/// works in a standalone checkout. With `STOCKCRAB_REQUIRE_CPP_SRC` set, a
/// missing source is a failure instead; set it only where the reference is
/// checked out (CI does so next to `STOCKFISH_DIR`). Point `STOCKFISH_DIR` at
/// a Stockfish checkout (the directory that contains `src/`).
// Not every test binary that includes this shared module calls this helper.
#[allow(dead_code)]
pub fn cpp_source_missing(test: &str, what: &str) {
    if std::env::var_os("STOCKCRAB_REQUIRE_CPP_SRC").is_some() {
        panic!(
            "{test}: the C++ Stockfish source ({what}) is required but missing (set STOCKFISH_DIR)"
        );
    }
    eprintln!("SKIP {test}: C++ Stockfish source ({what}) not found (set STOCKFISH_DIR)");
}

/// True when the default NNUE network is not where the engine looks for it
/// (the working directory or next to the test binary), in which case the
/// caller must return early: constructing an `Engine` without a network makes
/// `Engine::verify_network` exit the whole test process, as upstream does.
///
/// Prints the skip notice itself (or panics under `STOCKCRAB_REQUIRE_NET`, via
/// [`network_missing`]), so the usual form is
/// `if common::skip_without_network("name") { return; }`.
// Not every test binary that includes this shared module calls this helper.
#[allow(dead_code)]
pub fn skip_without_network(test: &str) -> bool {
    let name = stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME;
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join(name)));
    if std::path::Path::new(name).is_file() || beside_exe.is_some_and(|p| p.is_file()) {
        return false;
    }
    network_missing(test);
    true
}
