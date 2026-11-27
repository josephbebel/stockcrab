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

//! Preconditions that C++ only asserts are checked at the public boundary, so a wrong call
//! panics instead of reaching undefined behaviour in a release build: the check state each move
//! generator requires, and a perft depth within the position's state history.

use std::panic::{AssertUnwindSafe, catch_unwind};

use stockcrab::movegen::{
    GenType, MoveList, generate, generate_captures, generate_evasions, generate_quiets,
};
use stockcrab::position::Position;

fn pos(fen: &str) -> Position {
    let mut pos = Position::default();
    pos.set_fen(fen, false).unwrap();
    pos
}

#[test]
fn evasions_outside_check_panic() {
    let start = Position::default();
    assert!(
        catch_unwind(AssertUnwindSafe(|| generate_evasions(
            &start,
            &mut MoveList::new()
        )))
        .is_err()
    );
    assert!(catch_unwind(AssertUnwindSafe(|| generate(&start, GenType::Evasions))).is_err());
}

#[test]
fn captures_and_quiets_in_check_panic() {
    let checked = pos("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
    assert!(
        catch_unwind(AssertUnwindSafe(|| generate_captures(
            &checked,
            &mut MoveList::new()
        )))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| generate_quiets(
            &checked,
            &mut MoveList::new()
        )))
        .is_err()
    );
    let mut list = MoveList::new();
    generate_evasions(&checked, &mut list);
    assert_eq!(list.len(), 3, "Kxe2, Kd1 and Kf1");
}

#[test]
fn every_entry_point_agrees_with_generate_all() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    ] {
        let p = pos(fen);
        let all = generate(&p, GenType::All).len();
        let split = if p.checkers() != 0 {
            generate(&p, GenType::Evasions).len()
        } else {
            generate(&p, GenType::Captures).len() + generate(&p, GenType::Quiets).len()
        };
        assert_eq!(all, split, "{fen}");
    }
}

#[test]
fn perft_deeper_than_the_state_history_is_refused() {
    use stockcrab::types::MAX_HIST;
    let mut start = Position::default();
    let depth = MAX_HIST as i32;
    assert!(
        catch_unwind(AssertUnwindSafe(|| stockcrab::perft::perft(
            &mut start, depth
        )))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| stockcrab::perft::divide(
            &mut start, depth
        )))
        .is_err()
    );
    assert_eq!(stockcrab::perft::perft(&mut start, 3), 8902);
}

#[test]
fn zero_bitboard_scans_panic() {
    use stockcrab::bitboard::{lsb, msb, pop_lsb};

    assert!(std::panic::catch_unwind(|| lsb(0)).is_err());
    assert!(std::panic::catch_unwind(|| msb(0)).is_err());
    assert!(std::panic::catch_unwind(|| pop_lsb(&mut 0)).is_err());
}
