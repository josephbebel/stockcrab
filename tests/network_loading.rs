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

//! Strict and transactional NNUE loading through the public library interface.

use std::io::Write;
use stockcrab::nnue::network::Network;
use stockcrab::nnue::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use stockcrab::nnue::nnue_architecture::LAYER_STACKS;
use stockcrab::position::Position;

fn evaluations(network: &Network) -> [i32; LAYER_STACKS] {
    let mut position = Position::default();
    position.set_fen(stockcrab::uci::START_FEN, false).unwrap();
    let mut stack = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new();
    caches.clear(network);
    network
        .trace_evaluate(&position, &mut stack, &mut caches)
        .positional
}

fn layers(network: &Network) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..network.layer_stack_count() {
        network.network_arch().unwrap()[i]
            .write_parameters(&mut bytes)
            .unwrap();
    }
    bytes
}

#[test]
fn rejected_replacements_leave_all_active_weights_and_identity_unchanged() {
    stockcrab::bitboard::init();
    stockcrab::movegen::init();
    let original = std::fs::canonicalize(stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME)
        .expect("network loading regression requires scripts/fetch-net.sh");
    let mut network = Network::new();
    network
        .load("", original.to_string_lossy().into_owned())
        .unwrap();
    let expected_eval = evaluations(&network);
    let expected_layers = layers(&network);
    let expected_transformer = network.feature_transformer().unwrap() as *const _;
    let bytes = std::fs::read(&original).unwrap();
    let stacks = network.layer_stack_count();
    assert!(stacks > 1 && stacks <= LAYER_STACKS);
    let stack_size = expected_layers.len() / stacks + 4;
    let layers_start = bytes.len() - stack_size * stacks;
    // The layer-stack table follows the header (version, hash, description length and text):
    // its tag, the raw and stored stack counts, then one u16 fallback per raw configuration.
    let description_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let fallbacks_start = 12 + description_len + 12;
    let dir = tempfile::tempdir().unwrap();
    let bad_path = dir.path().join("rejected.nnue");

    for (label, length, patch, trailing) in [
        ("first stack truncated", layers_start + 4 + 16, None, false),
        (
            "last stack truncated",
            layers_start + (stacks - 1) * stack_size + 4 + 16,
            None,
            false,
        ),
        (
            "wrong layer hash",
            bytes.len(),
            Some((layers_start, bytes[layers_start] ^ 1)),
            false,
        ),
        // Configuration 0 falls back to a configuration out of range.
        (
            "invalid layer-stack fallback",
            bytes.len(),
            Some((fallbacks_start + 1, 0xFF)),
            false,
        ),
        ("trailing bytes", bytes.len(), None, true),
    ] {
        let mut file = std::fs::File::create(&bad_path).unwrap();
        file.write_all(&bytes[..length]).unwrap();
        if let Some((offset, byte)) = patch {
            use std::io::{Seek, SeekFrom};
            file.seek(SeekFrom::Start(offset as u64)).unwrap();
            file.write_all(&[byte]).unwrap();
        }
        if trailing {
            file.write_all(b"INVALID TRAILER").unwrap();
        }
        drop(file);
        assert!(
            network
                .load("", bad_path.to_string_lossy().into_owned())
                .is_err(),
            "{label}"
        );
        assert_eq!(
            network.feature_transformer().unwrap() as *const _,
            expected_transformer,
            "{label}"
        );
        assert_eq!(layers(&network), expected_layers, "{label}");
        assert_eq!(evaluations(&network), expected_eval, "{label}");
        // Verify identity without Network::verify, whose fatal path exits the
        // process; a repeated load must succeed and preserve the active weights.
        network
            .load("", original.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(evaluations(&network), expected_eval);
    }

    // A subsequent valid replacement must work after all failed transactions.
    std::fs::write(&bad_path, &bytes).unwrap();
    network
        .load("", bad_path.to_string_lossy().into_owned())
        .unwrap();
    assert_eq!(layers(&network), expected_layers);
    assert_eq!(evaluations(&network), expected_eval);
}
