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

//! End-to-end test that the NNUE weights are actually deduplicated across
//! processes.
//!
//! `tests/shm_sharing.rs` proves the shared-memory primitive works on an
//! arbitrary buffer. This proves the *network* is wired to it: a second engine
//! process must attach to the region published by the first rather than
//! allocating its own ~110 MiB copy.
//!
//! The distinguishing signal is `shared_weights_is_creator()`. Simply observing
//! "Shared memory." in both processes is not enough — two processes that each
//! created a private region would report exactly that.

mod common;

use std::process::Command;

use stockcrab::nnue::network::Network;
use stockcrab::shm::AllocationStatus;

const CHILD_ENV: &str = "STOCKCRAB_SHM_NET_CHILD";

/// Load the default network the same way the engine does.
fn load_default_network() -> Option<Network> {
    let mut network = Network::new();
    // The upstream `make net` target stores the default network in ../src.
    // Keep the crate-local paths as fallbacks for standalone test fixtures.
    for dir in ["../src", "", "."] {
        if network
            .load(dir, stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME.to_string())
            .is_ok()
        {
            return Some(network);
        }
    }
    None
}

#[test]
fn second_process_attaches_to_the_first_networks_weights() {
    if std::env::var(CHILD_ENV).is_ok() {
        return; // child body lives in the sibling test
    }

    let Some(network) = load_default_network() else {
        common::network_missing(module_path!());
        return;
    };

    if network.shared_weights_status() != AllocationStatus::SharedMemory {
        eprintln!(
            "SKIP: shared memory unavailable, status = {:?}",
            network.shared_weights_status()
        );
        return;
    }

    // The region is keyed by the network content hash, so if another engine
    // process already holds the same weights this process attaches instead of
    // creating, and the creator check below does not apply.
    if network.shared_weights_is_creator() != Some(true) {
        eprintln!(
            "SKIP: weights region already held by another process (is_creator = {:?})",
            network.shared_weights_is_creator()
        );
        return;
    }

    // Hold the region (and its fd server) alive across the child's run.
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args([
            "--exact",
            "child_network_attaches",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .output()
        .expect("failed to spawn child process");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "child failed\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("CHILD_NETWORK_ATTACHED"),
        "child did not attach to the parent's weights\n--- stdout ---\n{stdout}\n\
         --- stderr ---\n{stderr}"
    );

    drop(network);
}

/// Child half of [`second_process_attaches_to_the_first_networks_weights`].
#[test]
fn child_network_attaches() {
    if std::env::var(CHILD_ENV).is_err() {
        return;
    }

    let network = load_default_network().expect("child could not load the network");

    assert_eq!(
        network.shared_weights_status(),
        AllocationStatus::SharedMemory,
        "child did not get a shared mapping for the weights"
    );
    assert_eq!(
        network.shared_weights_is_creator(),
        Some(false),
        "child allocated its own weight region instead of attaching to the parent's"
    );

    println!("CHILD_NETWORK_ATTACHED");
}
