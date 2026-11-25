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

//! Self-checking invariants for the board representation and move generator.
//!
//! These tests assert *relationships* rather than golden numbers. Golden values
//! go stale with every upstream tuning change, whereas an invariant such as
//! "undo_move restores the position exactly" stays true across syncs and needs
//! no C++ engine to check. The tests target state-restoration failures (stale
//! `StateInfo` slots, castling rights), en passant squares kept when the
//! capture is illegal (which alters the zobrist key), and malformed boards.

use stockcrab::bitboard;
use stockcrab::movegen::{GenType, generate};
use stockcrab::position::Position;
use stockcrab::types::{Color, Move, PieceType, Square};

/// Positions spanning the features that historically broke: castling, en
/// passant, promotions, checks, and endgames.
const CORPUS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "2r3k1/pp2ppbp/3p1np1/q7/2P1P3/1PN1BP2/P2Q2PP/3R1RK1 b - - 0 1",
    "4rrk1/pp1n1pp1/2pb1q1p/3p4/3P4/1P1B1N1P/P1PQ1PP1/2KR3R w - - 0 1",
    "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
    "8/8/8/8/8/4k3/8/R3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
];

fn init() {
    bitboard::init();
    stockcrab::movegen::init();
}

fn setup(fen: &str) -> Position {
    let mut pos = Position::default();
    pos.set_fen(fen, false)
        .unwrap_or_else(|e| panic!("test FEN rejected: {fen}: {}", e.message()));
    pos
}

/// A complete, comparable snapshot of everything a move must restore.
#[derive(PartialEq, Eq, Debug)]
struct Snapshot {
    fen: String,
    key: u64,
    material_key: u64,
    checkers: u64,
    threats: [u64; 7],
    blockers: [u64; 2],
    pinners: [u64; 2],
    occupied: u64,
    white: u64,
    black: u64,
    ep: Option<Square>,
    rule50: i32,
}

fn snapshot(pos: &Position) -> Snapshot {
    Snapshot {
        fen: pos.fen(),
        key: pos.key(),
        material_key: pos.material_key(),
        checkers: pos.checkers(),
        threats: [
            stockcrab::types::ALL_PIECES,
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ]
        .map(|pt| pos.threats_by(pt)),
        blockers: [Color::White, Color::Black].map(|c| pos.blockers_for_king(c)),
        pinners: [Color::White, Color::Black].map(|c| pos.pinners(c)),
        occupied: pos.pieces(),
        white: pos.pieces_of_color(Color::White),
        black: pos.pieces_of_color(Color::Black),
        ep: pos.ep_square(),
        rule50: pos.rule50_count(),
    }
}

/// `undo_move` must restore the position *exactly*, for every legal move of
/// every corpus position.
///
/// A failure to restore state (for example a stale `StateInfo` slot
/// corrupting castling rights) is only caught by comparing the full position
/// before and after.
#[test]
fn undo_move_restores_every_field() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);
        let before = snapshot(&pos);

        for &mv in generate(&pos, GenType::All).as_slice() {
            // SAFETY: the move is legal.
            unsafe { pos.do_move(mv) };
            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };

            let after = snapshot(&pos);
            assert_eq!(
                before, after,
                "do_move/undo_move of {mv} did not restore the position\nFEN: {fen}"
            );
        }
    }
}

/// The same, two plies deep: a single-ply check can miss state that is clobbered
/// only once a second `StateInfo` is used.
#[test]
fn undo_move_restores_after_two_plies() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);
        let before = snapshot(&pos);

        for &m1 in generate(&pos, GenType::All).as_slice() {
            // SAFETY: the move is legal.
            unsafe { pos.do_move(m1) };

            let replies = generate(&pos, GenType::All);
            for &m2 in replies.as_slice().iter().take(4) {
                // SAFETY: the move is legal.
                unsafe { pos.do_move(m2) };
                // SAFETY: the move is legal.
                unsafe { pos.undo_move(m2) };
            }

            // SAFETY: the move is legal.
            unsafe { pos.undo_move(m1) };
            assert_eq!(
                before,
                snapshot(&pos),
                "two-ply do/undo starting with {m1} did not restore the position\nFEN: {fen}"
            );
        }
    }
}

/// The incrementally maintained zobrist key must equal the key computed from
/// scratch for the resulting position.
///
/// An incremental update that drifts from the recomputed value corrupts the
/// transposition table without any visible symptom.
#[test]
fn incremental_zobrist_key_matches_recomputed() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);

        for &mv in generate(&pos, GenType::All).as_slice() {
            // SAFETY: the move is legal.
            unsafe { pos.do_move(mv) };

            let incremental = pos.key();
            // Re-derive from the resulting FEN: a fresh parse recomputes the key
            // from nothing, so agreement means the incremental path is sound.
            let recomputed = setup(&pos.fen()).key();

            assert_eq!(
                incremental,
                recomputed,
                "zobrist key drifted after {mv}\nfrom: {fen}\nto:   {}",
                pos.fen()
            );

            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };
        }
    }
}

#[test]
fn null_move_round_trips() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);
        if pos.checkers() != 0 {
            continue; // null moves are illegal in check
        }
        let before = snapshot(&pos);

        // SAFETY: the move is legal.
        unsafe { pos.do_null_move() };
        // SAFETY: the move is legal.
        unsafe { pos.undo_null_move() };

        assert_eq!(
            before,
            snapshot(&pos),
            "null move did not round-trip\nFEN: {fen}"
        );
    }
}

/// No generated move may ever capture a king, in any generator.
///
/// A violation means the legality logic is broken, since a king capture would
/// remove a king from the board.
#[test]
fn no_generator_ever_captures_a_king() {
    init();

    for fen in CORPUS {
        let pos = setup(fen);

        // Only EVASIONS is valid in check, and only the others out of check;
        // `generate_all` asserts exactly this.
        let types: &[GenType] = if pos.checkers() != 0 {
            &[GenType::Evasions, GenType::All]
        } else {
            &[GenType::Captures, GenType::Quiets, GenType::All]
        };

        for &gt in types {
            for &mv in generate(&pos, gt).as_slice() {
                let victim = pos.piece_on(mv.to_sq());
                assert_ne!(
                    stockcrab::types::type_of(victim),
                    PieceType::King,
                    "{gt:?} generated {mv}, capturing a king\nFEN: {fen}"
                );
            }
        }
    }
}

/// Every legal move must leave the mover's own king unattacked -- checked by
/// actually playing it, not by trusting the generator's own logic.
#[test]
fn legal_moves_never_leave_own_king_attacked() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);

        for &mv in generate(&pos, GenType::All).as_slice() {
            let us = pos.side_to_move();
            // SAFETY: the move is legal.
            unsafe { pos.do_move(mv) };

            let ksq = pos.square_of(us, PieceType::King);
            let attacked = pos.attackers_to_exist(ksq, pos.pieces(), !us);
            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };

            assert!(
                !attacked,
                "legal move {mv} leaves the {us:?} king attacked\nFEN: {fen}"
            );
        }
    }
}

/// Cached threats must match attacks with the defending king removed, including after moves.
#[test]
fn cached_threats_follow_position_changes() {
    init();
    fn check(pos: &Position) {
        let us = pos.side_to_move();
        let occupied = pos.pieces() ^ bitboard::square_bb_pub(pos.square_of(us, PieceType::King));
        let them = pos.pieces_of_color(!us);
        let mut expected = 0;
        for sq in 0..64 {
            let square = Square::new(sq).unwrap();
            if pos.attackers_to_occ(square, occupied) & them != 0 {
                expected |= bitboard::square_bb_pub(square);
            }
        }
        assert_eq!(
            pos.threats_by(stockcrab::types::ALL_PIECES),
            expected,
            "{}",
            pos.fen()
        );
    }
    for fen in CORPUS {
        let mut pos = setup(fen);
        check(&pos);
        check(&pos.clone());
        for mv in generate(&pos, GenType::All) {
            // SAFETY: a generated legal move.
            unsafe { pos.do_move(mv) };
            check(&pos);
            check(&pos.clone());
            // SAFETY: undo the most recent move.
            unsafe { pos.undo_move(mv) };
            check(&pos);
        }
        if pos.checkers() == 0 {
            // SAFETY: not in check.
            unsafe { pos.do_null_move() };
            check(&pos);
            // SAFETY: undo the live most recent null move.
            unsafe { pos.undo_null_move() };
            check(&pos);
        }
    }
}

/// The merged TT-move validator must agree with generation for normal and uncommon moves.
#[test]
fn legal_validator_matches_generation() {
    init();
    for fen in CORPUS {
        let pos = setup(fen);
        let legal = generate(&pos, GenType::All);
        for from in 0..64u8 {
            for to in 0..64u8 {
                let from = Square::new(from).unwrap();
                let to = Square::new(to).unwrap();
                let moves = [
                    Move::from_to(from, to),
                    Move::make_en_passant(from, to),
                    Move::make_castling(from, to),
                    Move::make_promotion(from, to, PieceType::Knight),
                    Move::make_promotion(from, to, PieceType::Bishop),
                    Move::make_promotion(from, to, PieceType::Rook),
                    Move::make_promotion(from, to, PieceType::Queen),
                ];
                for mv in moves {
                    assert_eq!(
                        pos.legal(mv),
                        legal.contains(mv),
                        "validator disagrees with generator for {mv:?} in {fen}"
                    );
                }
            }
        }
    }
}

/// Captures and Quiets must partition All exactly -- no move in both,
/// none missing.
#[test]
fn captures_and_quiets_partition_all() {
    init();

    for fen in CORPUS {
        let pos = setup(fen);
        if pos.checkers() != 0 {
            continue;
        }

        let mut split: Vec<Move> = generate(&pos, GenType::Captures).as_slice().to_vec();
        let quiets: Vec<Move> = generate(&pos, GenType::Quiets).as_slice().to_vec();

        for q in &quiets {
            assert!(
                !split.contains(q),
                "move {q} appears in both Captures and Quiets\nFEN: {fen}"
            );
        }
        split.extend(quiets);
        assert_eq!(
            split.as_slice(),
            generate(&pos, GenType::All).as_slice(),
            "All must preserve capture-then-quiet order in {fen}"
        );
        split.sort_by_key(|m| m.raw());

        let mut all: Vec<Move> = generate(&pos, GenType::All).as_slice().to_vec();
        all.sort_by_key(|m| m.raw());

        assert_eq!(split, all, "Captures + Quiets != All\nFEN: {fen}");
    }
}

/// `pos_is_ok` must hold after every legal move.
///
/// It is a `debug_assert` inside the engine, so it does nothing in the release
/// builds this project normally tests. Calling it explicitly makes it real.
#[test]
fn position_stays_consistent_after_every_move() {
    init();

    for fen in CORPUS {
        let mut pos = setup(fen);

        for &mv in generate(&pos, GenType::All).as_slice() {
            // SAFETY: the move is legal.
            unsafe { pos.do_move(mv) };
            let ok = pos.pos_is_ok();
            let after = pos.fen();
            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };

            assert!(
                ok,
                "pos_is_ok() false after {mv}\nfrom: {fen}\nto:   {after}"
            );
        }
    }
}

/// Number of `CORPUS` positions `flip()` may reject without failing the mirror
/// tests below. Every corpus FEN currently mirrors to a valid position, so any
/// rejection is a regression; the guard keeps the tests from passing vacuously.
const MAX_UNMIRRORABLE: usize = 0;

/// `flip()` mirrors the board and swaps colours, so it must be an involution
/// and must preserve the number of legal moves.
#[test]
fn flip_is_an_involution_preserving_mobility() {
    init();

    let mut compared = 0;

    for fen in CORPUS {
        let mut pos = setup(fen);
        let original = pos.fen();
        let moves_before = generate(&pos, GenType::All).len();

        if pos.flip().is_some() {
            continue; // flipping produced a position the validator rejects
        }
        let moves_flipped = generate(&pos, GenType::All).len();

        assert_eq!(
            moves_before, moves_flipped,
            "flip changed the number of legal moves\nFEN: {fen}"
        );

        if pos.flip().is_some() {
            panic!("flipping back was rejected\nFEN: {fen}");
        }
        assert_eq!(pos.fen(), original, "flip is not an involution\nFEN: {fen}");
        compared += 1;
    }

    assert!(
        compared + MAX_UNMIRRORABLE >= CORPUS.len(),
        "only {compared} of {} positions were actually flipped and compared",
        CORPUS.len()
    );
}

/// perft must be invariant under mirroring: flipping the board vertically and
/// swapping colours produces a position with exactly the same move tree.
///
/// This is the strongest movegen check available without golden values, and it
/// is orthogonal to `tests/perft_extended.rs`: golden counts catch "wrong for
/// everyone", mirror symmetry catches "wrong for one colour" -- a bug class that
/// pawn direction, castling side and promotion rank handling are all prone to.
#[test]
fn perft_is_invariant_under_mirroring() {
    init();

    let mut compared = 0;

    for fen in CORPUS {
        let mut pos = setup(fen);
        let original = perft(&mut pos, 3);

        if pos.flip().is_some() {
            continue; // the mirrored position is not one the validator accepts
        }
        let mirrored_fen = pos.fen();
        let mirrored = perft(&mut pos, 3);

        assert_eq!(
            original, mirrored,
            "perft(3) differs between a position and its mirror\n\
             original: {fen}\n\
             mirrored: {mirrored_fen}"
        );
        compared += 1;
    }

    // Without this the test would pass vacuously if `flip` began failing for
    // every position.
    assert!(
        compared + MAX_UNMIRRORABLE >= CORPUS.len(),
        "only {compared} of {} positions were actually mirrored and compared",
        CORPUS.len()
    );
}

/// Simple recursive perft, independent of the engine's own perft entry point so
/// that a bug there cannot mask a movegen bug (or vice versa).
fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate(pos, GenType::All);
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for &mv in moves.as_slice() {
        // SAFETY: the move is legal.
        unsafe { pos.do_move(mv) };
        nodes += perft(pos, depth - 1);
        // SAFETY: the move is legal.
        unsafe { pos.undo_move(mv) };
    }
    nodes
}

/// The engine's own perft must agree with the independent one above.
///
/// They share the move generator but not the traversal, so a disagreement
/// isolates the bug to bulk-counting or state handling in `perft.rs`.
#[test]
fn engine_perft_agrees_with_a_naive_traversal() {
    init();

    for fen in CORPUS {
        let mut a = setup(fen);
        let mut b = setup(fen);

        for depth in 1..=3 {
            assert_eq!(
                stockcrab::perft::perft(&mut a, depth as i32),
                perft(&mut b, depth),
                "engine perft({depth}) disagrees with a naive traversal\nFEN: {fen}"
            );
        }
    }
}
