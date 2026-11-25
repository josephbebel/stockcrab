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

//! NNUE accumulator invariants.
//!
//! The `eval` command exercises only the NNUE **full-refresh** path, so it can
//! pass while the **incremental** path used by search is broken (for example
//! `fc_0` outputs not computed, or threat features not updated incrementally).
//! These tests compare both paths.
//!
//! The invariant is simple and needs no C++ engine: evaluating a position
//! reached by playing moves (incremental updates) must give exactly the same
//! number as evaluating that position from scratch (full refresh). Any drift in
//! the incremental path shows up immediately.
//!
//! What is covered:
//!
//! * `check_line` and the first-move test evaluate after *every* move, so the
//!   accumulator stack never has more than one uncomputed state: they cover the
//!   single-ply forward update and the refresh paths, but not a gap.
//! * `random_walk_matches_full_refresh` is the multi-ply test. One stack and one
//!   warm cache serve a long random sequence of moves, take-backs, evaluations
//!   and stack resets, so evaluations happen with several uncomputed states
//!   (forward update across a gap, `find_last_usable_accumulator`, the lagging
//!   perspective catch-up), after a king move inside a gap (refresh, then the
//!   backward update of the skipped states, which are checked when the walk
//!   returns to them) and over stale cache entries, in standard and Chess960
//!   positions.
//! * Colour symmetry and push/pop round trips.

mod common;

use stockcrab::bitboard;
use stockcrab::movegen::{GenType, generate};
use stockcrab::nnue::network::Network;
use stockcrab::nnue::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use stockcrab::position::Position;
use stockcrab::types::{Dirties, Move, MoveType, Piece};

const CORPUS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2r3k1/pp2ppbp/3p1np1/q7/2P1P3/1PN1BP2/P2Q2PP/3R1RK1 b - - 0 1",
    "4rrk1/pp1n1pp1/2pb1q1p/3p4/3P4/1P1B1N1P/P1PQ1PP1/2KR3R w - - 0 1",
    // King moves across the board centre force a HalfKA refresh; king moves that
    // stay on one side take the cheaper "hybrid" path. Both must agree.
    "8/8/8/8/8/4k3/8/4K2R w K - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    // Promotion and capture-promotion, and a legal en-passant capture.
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
];

/// Chess960 starting and middlegame positions (castling rights in X-FEN / Shredder notation).
const CORPUS_960: &[&str] = &[
    "rknrnbbq/pppppppp/8/8/4P3/8/PPPP1PPP/RKNRNBBQ w DAda - 0 1",
    "r1kbbq1r/3pp1p1/n4pp1/Npp4n/1p3P2/3P3P/P1P1P3/1RKBBQNR b HBh - 1 13",
    "nr1k1brq/nppbp1pp/5p2/p2p4/5N2/1N5P/PPPPPPPQ/2KRBBR1 b gb - 2 6",
    "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
];

fn init() {
    bitboard::init();
    stockcrab::movegen::init();
}

fn load_network() -> Option<Network> {
    let mut network = Network::new();
    // Same places the engine looks: the working directory, then the directory
    // holding the test executable.
    let mut dirs = vec![".".to_string()];
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.to_string_lossy().into_owned()))
    {
        dirs.push(exe_dir);
    }
    let mut load_error = None;
    for dir in &dirs {
        let name = stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME;
        if !std::path::Path::new(dir).join(name).is_file() {
            continue;
        }
        match network.load(dir, name.to_string()) {
            Ok(()) => return Some(network),
            Err(e) => load_error = Some(e),
        }
    }
    if let Some(e) = load_error {
        panic!("network found but failed to load: {e}");
    }
    None
}

fn setup(fen: &str) -> Position {
    let mut pos = Position::default();
    pos.set_fen(fen, false).expect("valid test FEN");
    pos
}

/// Evaluate `pos` with a brand-new accumulator stack, i.e. via a full refresh.
fn eval_from_scratch(network: &Network, pos: &Position) -> i32 {
    let mut stack = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new();
    caches.clear(network);
    stack.reset();
    network.evaluate(pos, &mut stack, &mut caches)
}

/// Play up to `depth` plies from `fen`, choosing a deterministic legal move at
/// each ply (by position key), updating the accumulator incrementally exactly as
/// the search does, and compare each step against a full refresh of the same
/// position.
fn check_line(network: &Network, fen: &str, depth: usize) {
    let mut pos = setup(fen);

    let mut stack = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new();
    caches.clear(network);
    stack.reset();

    // Baseline: the root itself must match.
    let root_inc = network.evaluate(&pos, &mut stack, &mut caches);
    let root_ref = eval_from_scratch(network, &pos);
    assert_eq!(
        root_inc, root_ref,
        "root eval differs from a full refresh\nFEN: {fen}"
    );

    let mut played: Vec<stockcrab::types::Move> = Vec::new();

    for _ in 0..depth {
        let moves = generate(&pos, GenType::All);
        if moves.is_empty() {
            break;
        }
        // Deterministic pick that still varies with the position.
        let mv = moves.as_slice()[(pos.key() as usize) % moves.len()];

        let gives_check = pos.gives_check(mv);
        // SAFETY: the pushed `Dirties` is filled by `do_move_internal` right below, before the
        // next evaluation.
        let dirties: *mut Dirties = unsafe { stack.push(&pos) };
        // SAFETY: `dirties` points into the freshly pushed accumulator state,
        // which lives in `stack` and is neither moved nor dropped during the call.
        unsafe {
            pos.do_move_internal(mv, gives_check, &mut *dirties);
        }
        played.push(mv);

        let incremental = network.evaluate(&pos, &mut stack, &mut caches);
        let refreshed = eval_from_scratch(network, &pos);

        assert_eq!(
            incremental,
            refreshed,
            "incremental accumulator diverged from a full refresh\n\
             start FEN : {fen}\n\
             line      : {}\n\
             now       : {}",
            played
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            pos.fen()
        );
    }
}

#[test]
fn incremental_accumulator_matches_full_refresh() {
    init();
    let Some(network) = load_network() else {
        common::network_missing(module_path!());
        return;
    };

    for fen in CORPUS {
        check_line(&network, fen, 24);
    }
}

/// The same, but exploring *every* first move rather than one line, so that
/// captures, promotions, castling and en passant are all covered at least once.
#[test]
fn incremental_accumulator_matches_refresh_for_every_first_move() {
    init();
    let Some(network) = load_network() else {
        common::network_missing(module_path!());
        return;
    };

    for fen in CORPUS {
        let mut pos = setup(fen);
        let moves: Vec<_> = generate(&pos, GenType::All).as_slice().to_vec();

        for mv in moves {
            let mut stack = AccumulatorStack::new();
            let mut caches = AccumulatorCaches::new();
            caches.clear(&network);
            stack.reset();
            let _ = network.evaluate(&pos, &mut stack, &mut caches);

            let gives_check = pos.gives_check(mv);
            // SAFETY: the pushed `Dirties` is filled by `do_move_internal` right below, before the
            // next evaluation.
            let dirties: *mut Dirties = unsafe { stack.push(&pos) };
            // SAFETY: `dirties` points at the entry just pushed on `stack`, which is neither moved nor dropped during the call.
            unsafe {
                pos.do_move_internal(mv, gives_check, &mut *dirties);
            }

            let incremental = network.evaluate(&pos, &mut stack, &mut caches);
            let refreshed = eval_from_scratch(&network, &pos);
            let after = pos.fen();
            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };

            assert_eq!(
                incremental, refreshed,
                "incremental update wrong for {mv}\nfrom: {fen}\nto:   {after}"
            );
        }
    }
}

/// Evaluation must be colour-symmetric: mirroring the board and swapping sides
/// negates the score.
///
/// This catches a whole class of perspective bugs -- a feature indexed from the
/// wrong king square, or one perspective updated while the other is not -- that
/// an absolute-value check cannot see. It is also exactly what the "update both
/// perspectives together" optimisation could have broken.
#[test]
fn evaluation_is_colour_symmetric() {
    init();
    let Some(network) = load_network() else {
        common::network_missing(module_path!());
        return;
    };

    let mut compared = 0;
    for fen in CORPUS {
        let mut pos = setup(fen);
        if pos.checkers() != 0 {
            continue;
        }
        let original = eval_from_scratch(&network, &pos);

        if pos.flip().is_some() {
            continue; // the mirrored position is not one the validator accepts
        }
        if pos.checkers() != 0 {
            continue;
        }
        let flipped = eval_from_scratch(&network, &pos);

        // Both are from the side-to-move's point of view, and flipping swaps who
        // is to move, so the scores must be equal, not negated.
        assert_eq!(
            original,
            flipped,
            "evaluation is not colour-symmetric\nFEN: {fen}\nflipped: {}",
            pos.fen()
        );
        compared += 1;
    }
    assert!(
        compared >= CORPUS.len(),
        "only {compared} of {} positions were compared",
        CORPUS.len()
    );
}

/// Popping back to an earlier accumulator must reproduce the earlier evaluation.
#[test]
fn accumulator_stack_pop_restores_earlier_evaluation() {
    init();
    let Some(network) = load_network() else {
        common::network_missing(module_path!());
        return;
    };

    for fen in CORPUS {
        let mut pos = setup(fen);
        let mut stack = AccumulatorStack::new();
        let mut caches = AccumulatorCaches::new();
        caches.clear(&network);
        stack.reset();

        let before = network.evaluate(&pos, &mut stack, &mut caches);

        for &mv in generate(&pos, GenType::All).as_slice().iter().take(8) {
            let gives_check = pos.gives_check(mv);
            // SAFETY: the pushed `Dirties` is filled by `do_move_internal` right below, before the
            // next evaluation.
            let dirties: *mut Dirties = unsafe { stack.push(&pos) };
            // SAFETY: `dirties` points at the entry just pushed on `stack`, which is neither moved nor dropped during the call.
            unsafe {
                pos.do_move_internal(mv, gives_check, &mut *dirties);
            }
            let _ = network.evaluate(&pos, &mut stack, &mut caches);

            // SAFETY: the move is legal.
            unsafe { pos.undo_move(mv) };
            stack.pop();

            let after = network.evaluate(&pos, &mut stack, &mut caches);
            assert_eq!(
                before, after,
                "evaluation changed after push/pop around {mv}\nFEN: {fen}"
            );
        }
    }
}

/// Deterministic xorshift64 generator for the random walk below.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Default)]
struct WalkStats {
    moves: usize,
    undos: usize,
    evals: usize,
    resets: usize,
    /// Evaluations made with at least three uncomputed states on the stack.
    gap_evals: usize,
    /// Evaluations right after a take-back (the state is then often one that a backward update
    /// or an earlier catch-up computed).
    evals_after_undo: usize,
    king_moves_cross_half: usize,
    king_moves_same_half: usize,
}

/// Random sequence of moves (without evaluating), take-backs, evaluations and stack resets on
/// one `AccumulatorStack` and one `AccumulatorCaches` that stay alive (and warm) for the whole
/// run. Every evaluation is compared with a from-scratch evaluation (new stack, new cache).
fn random_walk(
    network: &Network,
    caches: &mut AccumulatorCaches,
    rng: &mut XorShift,
    fen: &str,
    chess960: bool,
    iterations: usize,
    stats: &mut WalkStats,
) {
    /// Stays well below the stack capacity (`MAX_PLY + 1` states).
    const MAX_DEPTH: usize = 64;

    let mut pos = Position::default();
    pos.set_fen(fen, chess960).expect("valid test FEN");

    let mut stack = AccumulatorStack::new();
    stack.reset();

    let mut played: Vec<Move> = Vec::new();
    // Moves before this index were made before the last `stack.reset()`: the stack no longer has
    // their states, so they cannot be taken back.
    let mut floor = 0usize;
    // Plies made since the last evaluation / reset, and whether the last action was a take-back.
    let mut unevaluated = 0usize;
    let mut just_undid = false;

    for _ in 0..iterations {
        let describe = |pos: &Position, played: &[Move]| {
            format!(
                "start FEN: {fen} (chess960: {chess960})\nmoves    : {}\nnow      : {}",
                played
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                pos.fen()
            )
        };

        let depth = played.len() - floor;
        let mut action = rng.below(100);
        let moves = generate(&pos, GenType::All);
        // A dead end (mate, stalemate) or the depth limit forces a take-back, or ends the walk
        // when nothing can be taken back either.
        if (moves.is_empty() || depth >= MAX_DEPTH) && action < 48 {
            if depth == 0 {
                break;
            }
            action = 48;
        }

        match action {
            // Play a random legal move without evaluating.
            0..48 => {
                let moves = moves.as_slice();
                let kings: Vec<Move> = moves
                    .iter()
                    .copied()
                    .filter(|&m| matches!(pos.moved_piece(m), Piece::W_King | Piece::B_King))
                    .collect();
                // Favour king moves: they are the ones that force refreshes.
                let mv = if !kings.is_empty() && rng.below(100) < 30 {
                    kings[rng.below(kings.len())]
                } else {
                    moves[rng.below(moves.len())]
                };
                if matches!(pos.moved_piece(mv), Piece::W_King | Piece::B_King)
                    && mv.move_type() == MoveType::Normal
                {
                    if (mv.from_sq() as u8 & 4) == (mv.to_sq() as u8 & 4) {
                        stats.king_moves_same_half += 1;
                    } else {
                        stats.king_moves_cross_half += 1;
                    }
                }

                let gives_check = pos.gives_check(mv);
                // SAFETY: the pushed `Dirties` is filled by `do_move_internal` right below,
                // before the next evaluation.
                let dirties: *mut Dirties = unsafe { stack.push(&pos) };
                // SAFETY: `dirties` points into the freshly pushed accumulator state, which
                // lives in `stack` and is neither moved nor dropped during the call.
                unsafe {
                    pos.do_move_internal(mv, gives_check, &mut *dirties);
                }
                played.push(mv);
                stats.moves += 1;
                unevaluated += 1;
                just_undid = false;
            }
            // Take the last move back.
            48..68 => {
                if depth == 0 {
                    continue;
                }
                let mv = played.pop().unwrap();
                // SAFETY: `mv` is the last move made.
                unsafe { pos.undo_move(mv) };
                stack.pop();
                stats.undos += 1;
                unevaluated = unevaluated.saturating_sub(1);
                just_undid = true;
            }
            // Evaluate and compare with a full refresh.
            68..96 => {
                let incremental = network.evaluate(&pos, &mut stack, caches);
                let refreshed = eval_from_scratch(network, &pos);
                assert_eq!(
                    incremental,
                    refreshed,
                    "incremental evaluation differs from a full refresh\n{}",
                    describe(&pos, &played)
                );
                stats.evals += 1;
                stats.gap_evals += usize::from(unevaluated >= 3);
                stats.evals_after_undo += usize::from(just_undid);
                unevaluated = 0;
                just_undid = false;
            }
            // Restart the stack at the current position; the cache stays warm.
            _ => {
                stack.reset();
                floor = played.len();
                stats.resets += 1;
                unevaluated = 0;
                just_undid = false;
            }
        }
    }
}

/// Multi-ply, backward and catch-up accumulator updates against full refreshes; see the file
/// header. Deterministic (fixed seed), a few thousand steps over standard and Chess960 starts.
#[test]
fn random_walk_matches_full_refresh() {
    init();
    let Some(network) = load_network() else {
        common::network_missing(module_path!());
        return;
    };

    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
    let mut stats = WalkStats::default();
    // One cache for the whole test, as in a search thread: its entries go stale between walks.
    let mut caches = AccumulatorCaches::new();
    caches.clear(&network);

    for pass in 0..2 {
        for (i, fen) in CORPUS.iter().enumerate() {
            random_walk(&network, &mut caches, &mut rng, fen, false, 300, &mut stats);
            // Chess960 starts interleaved, in the same warm cache.
            if let Some(fen) = CORPUS_960.get(i + pass * 5) {
                random_walk(&network, &mut caches, &mut rng, fen, true, 300, &mut stats);
            }
        }
    }

    // The walk is only meaningful if the interesting situations were actually reached.
    assert!(stats.evals >= 500, "too few evaluations: {}", stats.evals);
    assert!(
        stats.gap_evals >= 50,
        "too few evaluations over a gap of 3+ plies: {}",
        stats.gap_evals
    );
    assert!(
        stats.evals_after_undo >= 50,
        "too few evaluations after a take-back"
    );
    assert!(stats.undos >= 500 && stats.resets >= 20);
    assert!(
        stats.king_moves_cross_half >= 20 && stats.king_moves_same_half >= 20,
        "too few king moves: {} across the d/e boundary, {} within a half",
        stats.king_moves_cross_half,
        stats.king_moves_same_half
    );
}
