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

//! Differential regression tests for FEN validation.
//!
//! `Position::set_fen` is a port of C++ `Position::set`, which strictly
//! validates its input and returns a `PositionSetError` instead of asserting.
//! Accepting a position that Stockfish rejects is not harmless: the engine may
//! then search a state its invariants do not cover (a side able to capture the
//! enemy king, a board with no king at all, ...), which is exactly the class of
//! bug `PositionSetError` exists to prevent.
//!
//! Every expectation below -- both the accepted positions and the exact error
//! text -- was produced by running the C++ reference engine built from the same
//! commit:
//!
//! ```text
//! printf 'position fen <FEN>\nquit\n' | ../src/stockfish | grep -oE 'Reason: .*'
//! ```
//!
//! so this file is a frozen snapshot of C++ behaviour rather than a record of
//! what stockcrab happened to do. If upstream changes its validation, re-derive
//! these the same way; do not relax an expectation to match stockcrab.
//!
//! The corpus mixes hand-written cases with survivors of a mutation fuzz over
//! valid FENs, and covers every distinct outcome the C++ parser can produce.

use stockcrab::bitboard;
use stockcrab::position::Position;

/// `(fen, expected)`, where `expected` is `None` for a position C++ accepts and
/// `Some(msg)` for the exact error message C++ reports.
const CASES: &[(&str, Option<&str>)] = &[
    (
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        None,
    ),
    ("2r5/8/8/8/8/8/8/K1k5 w - - 0 1", None),
    ("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2", None),
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3 w -", None),
    ("4k3/8/8/8/8/8/8/4K3 w - -", None),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0", None),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1 moves e1e2", None),
    ("4k3/8/8/8/8/8/8/4K3 w Aa - 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1", None),
    ("4k3/8/b7/2pP4/8/8/6K1/8 w - c6 0 2", None),
    ("8/8/8/8/8/8/8/K6k b - - 0 1", None),
    ("8/8/8/8/8/8/8/K6k w - - 0 1", None),
    ("K1k5/8/8/8/8/8/8/8 w - - 0 1", None),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", None),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", None),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0",
        None,
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        None,
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 32767 1",
        None,
    ),
    ("rrrrkrrr/8/8/8/8/8/8/4K3 w - - 0 1", None),
    (
        "4k3/8/8/2pP 4/8/8/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Board state encoding ended but cursor not at end."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0 1",
        Some("Invalid FEN. Board state encoding ended but cursor not at end."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkZ - 0 1",
        Some("Invalid FEN. Expected castling rights. Got: Z"),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkz - 0 1",
        Some("Invalid FEN. Expected castling rights. Got: z"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4K3 w- c6 0 2",
        Some("Invalid FEN. Expected whitespace after side to move."),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4K3 we- c6 08 2",
        Some("Invalid FEN. Expected whitespace after side to move."),
    ),
    (
        "4k3/8/8/8/8/8/8/4K3 b - e6 0 1",
        Some("Invalid FEN. Invalid en-passant square."),
    ),
    (
        "4k3/8/8/8/8/8/8/4K3 w - a1 0 1",
        Some("Invalid FEN. Invalid en-passant square."),
    ),
    (
        "4k3/8/8/8/8/8/8/4K3 w - e3 0 1",
        Some("Invalid FEN. Invalid en-passant square."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1",
        Some("Invalid FEN. Invalid en-passant square."),
    ),
    (
        "4k3/pppppppp1/8/8/8/8/8/4K3 w - - 0 1",
        Some("Invalid FEN. Invalid file reached."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1",
        Some("Invalid FEN. Invalid file reached."),
    ),
    (
        "04k3/8/8/2pP4/8/8/8/4K3 w - -6 02",
        Some("Invalid FEN. Invalid number of squares to skip."),
    ),
    (
        "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        Some("Invalid FEN. Invalid number of squares to skip."),
    ),
    (
        "-k3/8/8/2pP4/8/8/8/4K3 w - c6  2",
        Some("Invalid FEN. Invalid piece: -"),
    ),
    (
        "4k3/8/8/2pP-4/8/8/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: -"),
    ),
    (
        "./2p5/3p4/KP5r/1R3/1k/8/4P1P1/8 w - - 0 1",
        Some("Invalid FEN. Invalid piece: ."),
    ),
    (
        "4k3/.K/8/2pP4/8/8/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: ."),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4A3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: A"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4A3 w - c6h 0 2",
        Some("Invalid FEN. Invalid piece: A"),
    ),
    (
        "4k3/H/8/2pP4/8/8/8/4K3 w -xc6 0 2",
        Some("Invalid FEN. Invalid piece: H"),
    ),
    (
        "8/2Zp5/3p4/KP5r/1R3p1k/g8/4P1P1/8 w 4 - 0 1",
        Some("Invalid FEN. Invalid piece: Z"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4a3 w - cb 0 2",
        Some("Invalid FEN. Invalid piece: a"),
    ),
    (
        "4k3/8/8/2pPa4/8/8/8/4K3 w - c6 0 q2",
        Some("Invalid FEN. Invalid piece: a"),
    ),
    (
        "1rb3ck2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        Some("Invalid FEN. Invalid piece: c"),
    ),
    (
        "4cka3/8/8/2pP4/8/8/8/4K3 w K c6 0 2",
        Some("Invalid FEN. Invalid piece: c"),
    ),
    (
        "4e3/8/8/2pP4/k/8/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: e"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4eK3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: e"),
    ),
    (
        "4kg/8/82pP4/88/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: g"),
    ),
    (
        "43h/8/8/2pP4/8/8/8/4K3  - c6 0 2",
        Some("Invalid FEN. Invalid piece: h"),
    ),
    (
        "4kh/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
        Some("Invalid FEN. Invalid piece: h"),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/8 w KQkq - 0 1",
        Some("Invalid FEN. Invalid rank reached."),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4K3  - c6 0 2",
        Some("Invalid FEN. Invalid side to move: -"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4K3 2w - c6 0 2",
        Some("Invalid FEN. Invalid side to move: 2"),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4K3 Q - c6 0 2",
        Some("Invalid FEN. Invalid side to move: Q"),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        Some("Invalid FEN. Invalid side to move: x"),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkqK - 0 1",
        Some("Invalid FEN. Maximum of 4 castling rights can be specified."),
    ),
    (
        "rnbqkbnr/pppppppp/pppppppp/pppppppp/PPPPPPPP/PPPPPPPP/PPPPPPPP/RNBQKBNR w - - 0 1",
        Some("Invalid FEN. More than 32 pieces on the board."),
    ),
    (
        "/2p/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        Some("Invalid FEN. Trying to end rank when not at the end of it."),
    ),
    (
        "/2p0/3p4/KP5r/1R3p1k/8/4P21P1/8 w - - 0 1",
        Some("Invalid FEN. Trying to end rank when not at the end of it."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        Some("Invalid FEN. Unexpected end of stream."),
    ),
    (
        "4k3/pppppppp/pppppppp/8/8/8/8/4K3 w - - 0 1",
        Some("Unsupported position. BLACK has more than 8 pawns."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 999999",
        Some("Unsupported position. Game ply out of range."),
    ),
    (
        "4k3/8/8/8/8/8/8/3KK3 w - - 0 1",
        Some("Unsupported position. Incorrect number of kings."),
    ),
    (
        "5k2/7R/4Pr1p/8/p4P1p/8/8/8 w - - 0 4",
        Some("Unsupported position. Incorrect number of kings."),
    ),
    (
        "8/8/8/3Pp3/8/8/8/4K3 w - e6 0 2",
        Some("Unsupported position. Incorrect number of kings."),
    ),
    (
        "8/8/8/8/8/8/8/8 w - - 0 1",
        Some("Unsupported position. Incorrect number of kings."),
    ),
    (
        "4k3/8/8/8/8/8/4R3/R3K3 w - - 0 1",
        Some("Unsupported position. King can be captured."),
    ),
    (
        "4k3/8/8/2pP4/8/8/8/4P3 w - c6 0 2",
        Some("Unsupported position. Pawns on the first or eighth rank."),
    ),
    (
        "pppppppp/8/8/8/8/8/8/K6k w - - 0 1",
        Some("Unsupported position. Pawns on the first or eighth rank."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1",
        Some("Unsupported position. Rule50 counter out of range."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 32768 1",
        Some("Unsupported position. Rule50 counter out of range."),
    ),
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99999 1",
        Some("Unsupported position. Rule50 counter out of range."),
    ),
    (
        "bbbbkbbb/bbbbbbbb/8/8/8/8/8/4K3 w - - 0 1",
        Some("Unsupported position. Too many pieces for BLACK."),
    ),
    (
        "nnnnknnn/nnnnnnnn/8/8/8/8/8/4K3 w - - 0 1",
        Some("Unsupported position. Too many pieces for BLACK."),
    ),
    (
        "qqqqkqqq/qqqqqqqq/8/8/8/8/8/4K3 w - - 0 1",
        Some("Unsupported position. Too many pieces for BLACK."),
    ),
    (
        "4k3/8/8/8/8/8/QQQQQQQQ/QQQQQQQK w - - 0 1",
        Some("Unsupported position. Too many pieces for WHITE."),
    ),
    (
        "4k3/8/8/8/8/PPPPPPPP/PPPPPPPP/4K3 w - - 0 1",
        Some("Unsupported position. WHITE has more than 8 pawns."),
    ),
];

/// Rebuild a FEN the way `UciEngine::position` hands it to `set_fen`.
///
/// The UCI layer splits the command into whitespace-delimited tokens and
/// re-emits each one followed by a single space (C++ `fen += token + " "`).
/// That is not cosmetic: runs of whitespace collapse, and the string always
/// ends in a space -- which is what lets `set_fen` accept a FEN whose last
/// field is the side to move. The expectations below were captured through
/// that layer, so the test has to apply the same normalisation.
fn uci_normalize(fen: &str) -> String {
    fen.split_whitespace().map(|t| format!("{t} ")).collect()
}

fn set(fen: &str) -> Result<(), String> {
    let mut pos = Position::default();
    pos.set_fen(&uci_normalize(fen), false)
        .map(|_| ())
        .map_err(|e| e.message().to_string())
}

#[test]
fn fen_validation_matches_cpp() {
    bitboard::init();

    let mut failures = Vec::new();
    for &(fen, expected) in CASES {
        let got = set(fen);
        let ok = match (expected, &got) {
            (None, Ok(())) => true,
            (Some(want), Err(have)) => want == have,
            _ => false,
        };
        if !ok {
            let got = match got {
                Ok(()) => "(accepted)".to_string(),
                Err(e) => e,
            };
            failures.push(format!(
                "  FEN  {fen}\n    expected: {}\n    got:      {got}",
                expected.map_or("(accepted)".to_string(), str::to_string),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} FEN validation cases diverged from the C++ reference:\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n")
    );
}

/// Guards against the corpus silently losing coverage of an error kind, which
/// would let a whole validation branch rot untested. Messages that differ only
/// in the offending character (`Invalid piece: X`) count as one kind.
#[test]
fn corpus_covers_every_error_kind() {
    let kinds: std::collections::BTreeSet<&str> = CASES
        .iter()
        .filter_map(|&(_, e)| e)
        .map(|m| m.split(':').next().unwrap_or(m))
        .collect();
    assert_eq!(
        kinds.len(),
        22,
        "expected 22 distinct C++ error kinds, found {}: {:#?}",
        kinds.len(),
        kinds
    );
    assert!(
        CASES.iter().any(|&(_, e)| e.is_none()),
        "corpus must also contain positions that are accepted"
    );
}

/// The en passant square is only recorded when the capture is actually
/// available *and legal*. Keeping it otherwise changes the zobrist key and
/// would desynchronise the transposition table from C++.
#[test]
fn en_passant_square_is_dropped_when_unusable() {
    bitboard::init();

    let cases = [
        // No white pawn can capture on c6: ep square must be dropped.
        ("4k3/8/8/2p5/8/8/8/4K3 w - c6 0 2", false),
        // White pawn on d5 can capture c6 en passant, and doing so leaves the
        // white king safe: the square is kept.
        ("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2", true),
    ];

    for (fen, expect_ep) in cases {
        let mut pos = Position::default();
        pos.set_fen(fen, false)
            .unwrap_or_else(|e| panic!("{fen} should be accepted, got {}", e.message()));
        let has_ep = pos.ep_square().is_some();
        assert_eq!(has_ep, expect_ep, "ep handling wrong for {fen}");
    }
}
