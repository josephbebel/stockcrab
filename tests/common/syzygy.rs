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

//! Release prerequisites and coverage markers for the two Syzygy test targets.
//!
//! Direct required-mode tests validate all named WDL/DTZ headers. The release
//! runner additionally verifies every full SHA-256 before executing the suite.

use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

const MANIFEST: &str = include_str!("../fixtures/syzygy/3-4-5.sha256");

pub fn required() -> bool {
    std::env::var("STOCKCRAB_REQUIRE_SYZYGY").is_ok_and(|value| value == "1")
}

/// Check exact required filenames, both formats, magic and nontruncated lengths.
/// The caller provides the path, so negative tests need no process-global env edits.
pub fn validate_complete(path: &Path) -> Result<(), String> {
    for line in MANIFEST.lines().filter(|line| !line.starts_with('#')) {
        let Some(name) = line.split_whitespace().nth(1) else {
            continue;
        };
        let file = path.join(name);
        let mut input =
            std::fs::File::open(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        let len = input.metadata().map_err(|error| error.to_string())?.len();
        if len < 16 || len % 64 != 16 {
            return Err(format!("{}: invalid table length {len}", file.display()));
        }
        let mut magic = [0_u8; 4];
        input
            .read_exact(&mut magic)
            .map_err(|error| error.to_string())?;
        let expected = if name.ends_with("rtbw") {
            [0x71, 0xe8, 0x23, 0x5d]
        } else {
            [0xd7, 0x66, 0x0c, 0xa5]
        };
        if magic != expected {
            return Err(format!("{}: invalid table magic", file.display()));
        }
    }
    Ok(())
}

fn complete_path() -> Result<String, String> {
    // Test environments are fixed before subprocess creation, never mutated by
    // parallel tests. Cache the 290 header checks once in each integration target.
    static CHECK: OnceLock<Result<String, String>> = OnceLock::new();
    CHECK
        .get_or_init(|| {
            let path =
                std::env::var("SYZYGY_PATH").map_err(|_| "SYZYGY_PATH is not set".to_string())?;
            validate_complete(Path::new(&path))?;
            Ok(path)
        })
        .clone()
}

/// One marker per top-level test, including early returns and unwinding failures.
/// Position counters cover the selective probe loops, not every internal search node.
pub struct Coverage {
    name: &'static str,
    skipped: bool,
    checked: usize,
    skipped_cases: usize,
    failed_cases: usize,
}

impl Coverage {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            skipped: false,
            checked: 0,
            skipped_cases: 0,
            failed_cases: 0,
        }
    }

    pub fn tables(&mut self, full_set: bool) -> Option<String> {
        let result = if required() || full_set {
            complete_path()
        } else {
            std::env::var("SYZYGY_PATH")
                .map_err(|_| "SYZYGY_PATH is not set".to_string())
                .and_then(|path| {
                    if Path::new(&path).is_dir() {
                        Ok(path)
                    } else {
                        Err(format!("not a directory: {path}"))
                    }
                })
        };
        match result {
            Ok(path) => Some(path),
            Err(error) if required() => panic!(
                "{}: required 3–5 WDL+DTZ set unavailable: {error}",
                self.name
            ),
            Err(error) => {
                self.skipped = true;
                eprintln!("SKIP {}: {error}", self.name);
                None
            }
        }
    }

    pub fn network(&mut self) -> bool {
        if super::common::skip_without_network(self.name) {
            self.skipped = true;
            false
        } else {
            true
        }
    }

    /// A failed required <=5-piece probe is a regression, not a missing optional
    /// download. A present optional root table must also probe successfully.
    /// Six/seven-piece cases can still succeed through captures into the small set.
    /// `small_set_covers` marks cases whose success with that set is already
    /// confirmed by the pinned C++ engine, even if the root has more than five men.
    // The search target shares this helper but has no selectively probed positions.
    #[allow(dead_code)]
    pub fn probe_available(
        &mut self,
        path: &str,
        fen: &str,
        success: bool,
        small_set_covers: bool,
    ) -> bool {
        if success {
            self.checked += 1;
            return true;
        }
        let board = fen.split_whitespace().next().expect("FEN board");
        let mut white = String::new();
        let mut black = String::new();
        for piece in ['K', 'Q', 'R', 'B', 'N', 'P'] {
            white.extend(board.chars().filter(|&c| c == piece));
            black.extend(
                board
                    .chars()
                    .filter(|&c| c == piece.to_ascii_lowercase())
                    .map(|_| piece),
            );
        }
        let pieces = white.len() + black.len();
        let present = [
            format!("{white}v{black}.rtbw"),
            format!("{black}v{white}.rtbw"),
        ]
        .iter()
        .any(|name| Path::new(path).join(name).is_file());
        if (required() && (pieces <= 5 || small_set_covers)) || present {
            self.failed_cases += 1;
            panic!(
                "{}: probe failed for required or installed material: {fen}",
                self.name
            );
        }
        self.skipped_cases += 1;
        eprintln!(
            "STOCKCRAB_SYZYGY_CASE test={} status=skipped pieces={pieces} fen={fen}",
            self.name
        );
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_or_unrelated_files_do_not_satisfy_the_manifest() {
        let _coverage = Coverage::new(
            "syzygy_coverage::tests::absent_or_unrelated_files_do_not_satisfy_the_manifest",
        );
        let dir = tempfile::tempdir().unwrap();
        assert!(validate_complete(&dir.path().join("absent")).is_err());
        for i in 0..145 {
            std::fs::write(dir.path().join(format!("unrelated-{i}.rtbw")), [0_u8; 16]).unwrap();
        }
        assert!(validate_complete(dir.path()).is_err());
    }

    #[test]
    fn complete_headers_require_dtz_and_valid_lengths_and_magic() {
        let _coverage = Coverage::new(
            "syzygy_coverage::tests::complete_headers_require_dtz_and_valid_lengths_and_magic",
        );
        let dir = tempfile::tempdir().unwrap();
        let names: Vec<_> = MANIFEST
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| line.split_whitespace().nth(1))
            .collect();
        assert_eq!(names.len(), 290);
        for name in &names {
            let mut header = [0_u8; 16];
            header[..4].copy_from_slice(if name.ends_with("rtbw") {
                &[0x71, 0xe8, 0x23, 0x5d]
            } else {
                &[0xd7, 0x66, 0x0c, 0xa5]
            });
            std::fs::write(dir.path().join(name), header).unwrap();
        }
        // Structural check only: the Python release runner rejects these fake
        // bodies by SHA-256. This test does not claim they are usable tables.
        assert!(validate_complete(dir.path()).is_ok());
        let dtz = dir.path().join("KRvK.rtbz");
        std::fs::remove_file(&dtz).unwrap();
        assert!(validate_complete(dir.path()).is_err());
        std::fs::write(&dtz, [0_u8; 16]).unwrap();
        assert!(validate_complete(dir.path()).is_err());
        std::fs::write(&dtz, [0xd7, 0x66, 0x0c, 0xa5]).unwrap();
        assert!(validate_complete(dir.path()).is_err());
    }
}

impl Drop for Coverage {
    fn drop(&mut self) {
        let status = if std::thread::panicking() {
            "failed"
        } else if self.skipped {
            "skipped"
        } else {
            "passed"
        };
        eprintln!(
            "STOCKCRAB_SYZYGY test={} status={status} cases_checked={} cases_skipped={} cases_failed={}",
            self.name, self.checked, self.skipped_cases, self.failed_cases
        );
    }
}
