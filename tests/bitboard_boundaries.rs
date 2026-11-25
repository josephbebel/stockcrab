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

use std::num::NonZeroU64;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use stockcrab::bitboard::*;
use stockcrab::position::Position;
use stockcrab::types::{Color, PieceType, Square};

#[test]
fn empty_scans_fail_safely() {
    assert!(catch_unwind(|| lsb(0)).is_err());
    assert!(catch_unwind(|| msb(0)).is_err());
    let mut empty = 0;
    assert!(catch_unwind(AssertUnwindSafe(|| pop_lsb(&mut empty))).is_err());
    assert_eq!(empty, 0);
    for bit in 0..64 {
        let bb = 1u64 << bit;
        assert_eq!(lsb_nonzero(NonZeroU64::new(bb).unwrap()) as u8, bit);
        assert_eq!(msb(bb) as u8, bit);
        let mut remaining = bb;
        assert_eq!(pop_lsb(&mut remaining) as u8, bit);
        assert_eq!(remaining, 0);
    }
}

#[test]
fn positions_remain_valid_after_rejected_fen() {
    let mut pos = Position::default();
    assert_eq!(pos.fen(), stockcrab::uci::START_FEN);
    assert_eq!(pos.square_of(Color::White, PieceType::King), Square::SQ_E1);
    let fen = pos.fen();
    let key = pos.key();
    for invalid in ["bad fen", "8/8/8/8/8/8/8/8 w - - 0 1"] {
        assert!(pos.set_fen(invalid, false).is_err());
        assert_eq!(pos.fen(), fen);
        assert_eq!(pos.key(), key);
    }
    assert!(catch_unwind(|| pos.square_of(Color::White, PieceType::Queen)).is_ok());
    assert!(catch_unwind(|| pos.square_of(Color::White, PieceType::Pawn)).is_err());
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn cold_public_attack_tables() {
    const MODE: &str = "STOCKCRAB_COLD_BITBOARD";
    if let Ok(mode) = std::env::var(MODE) {
        let sq = Square::SQ_D4;
        match mode.as_str() {
            "rook" => assert_eq!(
                attacks_bb_rook(sq, 0), // first call, before any explicit init
                attacks_bb_scan(PieceType::Rook, sq, 0)
            ),
            "knight" => assert_eq!(attacks_bb_knight(sq).count_ones(), 8),
            "line" => assert_eq!(line_bb_pub(Square::SQ_A1, Square::SQ_H8).count_ones(), 8),
            "position" => assert_eq!(
                Position::default().square_of(Color::Black, PieceType::King),
                Square::SQ_E8
            ),
            "concurrent" => std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(move || {
                        assert_eq!(
                            attacks_bb_bishop(sq, 0),
                            attacks_bb_scan(PieceType::Bishop, sq, 0)
                        );
                        assert_eq!(
                            attacks_bb_rook(sq, 0),
                            attacks_bb_scan(PieceType::Rook, sq, 0)
                        );
                    });
                }
            }),
            _ => panic!("unknown mode"),
        }
        return;
    }
    for mode in ["rook", "knight", "line", "position", "concurrent"] {
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "cold_public_attack_tables", "--nocapture"])
                .env(MODE, mode)
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "{mode}: {status}");
                break;
            }
            assert!(Instant::now() < deadline, "{mode} hung");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
