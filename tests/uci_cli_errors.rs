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

//! CLI error handling and related input-parsing behaviour. Expected strings were taken from the C++ engine.
//!
//! None of these commands reaches the network or a search, so no network is needed.

use std::io::Write;
use std::process::{Command, Stdio};

/// Run the engine on `input` (raw bytes); returns (stdout, exit code).
fn run(input: &[u8]) -> (String, Option<i32>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_stockcrab"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engine");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(input)
        .expect("write");
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code(),
    )
}

/// Assert the session died with exit code 1 and the given CRITICAL ERROR line.
fn assert_critical(input: &str, command: &str, reason: &str) {
    let (stdout, code) = run(input.as_bytes());
    let expected =
        format!("info string CRITICAL ERROR: Command `{command}` failed. Reason: {reason}");
    assert!(
        stdout.lines().any(|l| l == expected),
        "input {input:?}: expected line {expected:?}, got stdout {stdout:?}"
    );
    assert_eq!(code, Some(1), "input {input:?}: exit code");
}

#[test]
fn go_with_invalid_argument_is_a_critical_error() {
    for (cmd, token) in [
        ("go depth 3 mate abc", "mate"),
        ("go depth 1 nodes", "nodes"),
        ("go depth 2 wtime e2e4", "wtime"),
        ("go depth 1 depth x", "depth"),
        ("go depth 99999999999", "depth"),
        ("go movetime", "movetime"),
        ("go perft -", "perft"),
    ] {
        assert_critical(
            &format!("{cmd}\nquit\n"),
            cmd,
            &format!("Invalid argument for '{token}'"),
        );
    }
}

#[test]
fn position_fen_with_missing_fields_is_a_critical_error() {
    for cmd in ["position fen", "position fen ", "position fen moves e2e4"] {
        // The command is echoed untrimmed, like C++.
        assert_critical(
            &format!("{cmd}\nquit\n"),
            cmd,
            "Invalid FEN. Unexpected end of stream.",
        );
    }
}

#[test]
fn fen_halfmove_clock_overflow_is_rejected() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99999999999999999999 1";
    let cmd = format!("position fen {fen}");
    assert_critical(
        &format!("{cmd}\nquit\n"),
        &cmd,
        "Unsupported position. Rule50 counter out of range.",
    );
    // An unparsable halfmove clock reads as 0 (valid) and sets failbit, so the
    // fullmove field (9) is never read; gamePly stays at its zeroed default and
    // the printed fullmove number is 1, as in C++.
    let (stdout, code) = run(b"position fen 4k3/8/8/8/8/8/8/4K3 b - - x 9\nd\nquit\n");
    assert_eq!(code, Some(0));
    assert!(
        stdout.contains("Fen: 4k3/8/8/8/8/8/8/4K3 b - - 0 1"),
        "{stdout}"
    );
}

#[test]
fn setoption_discards_its_first_token_like_cpp() {
    let (stdout, code) = run(
        b"setoption value 3\nsetoption\nsetoption name\nsetoption junk name Hash value 4\nquit\n",
    );
    assert_eq!(code, Some(0));
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("No such option"))
        .collect();
    assert_eq!(
        lines,
        [
            "No such option: 3",
            "No such option: ",
            "No such option: ",
            "No such option: name Hash"
        ]
    );
}

#[test]
fn invalid_utf8_does_not_end_the_session() {
    // C++ treats the line as raw bytes: it reports an unknown command (echoing the bytes) and
    // carries on with the next line.
    let mut input = b"foo\xff\xfe bar\n".to_vec();
    input.extend_from_slice(b"isready\nquit\n");
    let (stdout, code) = run(&input);
    assert_eq!(code, Some(0));
    assert!(stdout.contains("Unknown command: 'foo"), "{stdout}");
    assert!(
        stdout.lines().any(|l| l == "readyok"),
        "session ended early: {stdout:?}"
    );
}

#[test]
fn unknown_command_echoes_the_untrimmed_line() {
    let (stdout, _) = run(b"  hello   world  \nquit\n");
    assert!(
        stdout
            .lines()
            .any(|l| l == "Unknown command: '  hello   world  '. Type help for more information."),
        "{stdout:?}"
    );
}

#[test]
fn numa_policy_string_parsing_matches_cpp() {
    // (value, expected info string). Unparsable items are skipped, an empty result or a duplicate
    // CPU keeps the previous configuration.
    for (value, expected) in [
        ("0-3:4-7", "Available processors: 0-3:4-7"),
        ("0, 1", "Available processors: 0-1"),
        ("0,abc", "Available processors: 0"),
        ("1:", "Available processors: 1"),
        ("+1:2", "Available processors: 1:2"),
        (
            "abc",
            "NumaPolicy: invalid value 'abc', keeping previous config.",
        ),
        (
            "0,0",
            "NumaPolicy: invalid value '0,0', keeping previous config.",
        ),
        (
            "3-1",
            "NumaPolicy: invalid value '3-1', keeping previous config.",
        ),
        (
            "0-1048576",
            "NumaPolicy: invalid value '0-1048576', keeping previous config.",
        ),
    ] {
        let (stdout, code) =
            run(format!("setoption name NumaPolicy value {value}\nquit\n").as_bytes());
        assert_eq!(code, Some(0));
        let want = format!("info string {expected}");
        assert!(
            stdout.lines().any(|l| l == want),
            "value {value:?}: want {want:?}, got {stdout:?}"
        );
    }
}

#[test]
fn option_names_fold_ascii_case_only() {
    // C++ compares option names byte-wise with `tolower`; U+212A (Kelvin sign) must not fold to `k`.
    let (stdout, _) = run(
        "setoption name S\u{212A}ill Level value 3\nsetoption name SKILL level value 3\nquit\n"
            .as_bytes(),
    );
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("No such option"))
        .collect();
    assert_eq!(lines, ["No such option: S\u{212A}ill Level"]);
}

/// C++ `Network::save` opens the stream unchecked, so an unwritable path surfaces only as the
/// boolean result and the line `Failed to export a net` on stdout; nothing goes to stderr.
#[test]
fn export_net_to_an_unwritable_path_prints_failed_to_export_a_net() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("no-such-dir").join("x.nnue");
    let (stdout, code) = run(format!("export_net {}\nquit\n", target.display()).as_bytes());
    assert!(
        stdout.lines().any(|l| l == "Failed to export a net"),
        "stdout: {stdout:?}"
    );
    assert!(!target.exists());
    assert_eq!(code, Some(0));
}

/// C++ `Network::save` first checks that a network file is loaded
/// (`!evalFile.current.has_value()`), with or without a filename. Stockcrab embeds no network,
/// so the state is reached by running a copy of the binary from a directory without one. The
/// expected line is what a C++ build with `-DNNUE_EMBEDDING_OFF` prints there.
#[test]
fn export_net_without_a_loaded_network_prints_the_upstream_message() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let dir = tmp.path();
    let exe = dir.join("stockcrab-copy");
    std::fs::copy(env!("CARGO_BIN_EXE_stockcrab"), &exe).expect("copy engine");
    // A child forked by another test thread while `copy` still had the file open for writing
    // holds that descriptor until it execs, and until then exec fails with ETXTBSY. Retry.
    let mut attempts = 0;
    let mut child = loop {
        match Command::new(&exe)
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && attempts < 100 => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            spawned => break spawned.expect("failed to spawn engine"),
        }
    };
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(b"export_net out.nnue\nexport_net\nquit\n")
        .expect("write");
    let out = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let created = dir.join("out.nnue").exists();
    let entries = std::fs::read_dir(dir).map_or(0, Iterator::count);
    let failures: Vec<&str> = stdout.lines().skip(1).collect();
    let expected = "Failed to export a net. No network file is currently loaded. \
                    Please load a network file first.";
    assert_eq!(failures, [expected, expected], "stdout: {stdout:?}");
    assert!(!created);
    assert_eq!(entries, 1, "only the engine copy may be in the directory");
    assert_eq!(out.status.code(), Some(0));
}

/// C++ prints `sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl`, i.e. the count
/// is followed by a blank line.
#[test]
fn perft_count_is_followed_by_a_blank_line() {
    let (stdout, code) = run(b"position startpos\ngo perft 1\nquit\n");
    assert_eq!(code, Some(0));
    assert!(
        stdout.contains("\nNodes searched: 20\n\n"),
        "stdout: {stdout:?}"
    );
    assert!(
        !stdout.contains("Nodes searched: 20\n\n\n"),
        "stdout: {stdout:?}"
    );
}

/// The licence text is one block: a leading blank line, six contiguous lines, a trailing blank
/// line (uci.cpp `help`/`license`).
#[test]
fn help_prints_one_contiguous_block_with_a_trailing_blank_line() {
    let expected = "\nStockcrab is a Rust port of Stockfish for playing and analyzing chess.\n\
        It is released as free software licensed under the GNU GPLv3 License.\n\
        Stockcrab is normally used with a graphical user interface (GUI) and implements\n\
        the Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc.\n\
        For any further information, read the README.md and LICENSE files distributed with Stockcrab.\n\
        Upstream Stockfish: https://github.com/official-stockfish/Stockfish#readme\n\n";
    for cmd in ["help", "license", "--help", "--license"] {
        let (stdout, code) = run(format!("{cmd}\nquit\n").as_bytes());
        assert_eq!(code, Some(0));
        assert!(stdout.ends_with(expected), "{cmd}: stdout {stdout:?}");
    }
}

/// `compiler` is framed by a blank line before and after the block, as in C++.
#[test]
fn compiler_output_is_framed_by_blank_lines() {
    let (stdout, code) = run(b"compiler\nquit\n");
    assert_eq!(code, Some(0));
    let (_, block) = stdout
        .split_once("\nCompiled by")
        .expect("Compiled by line");
    assert!(block.ends_with("\n\n"), "stdout: {stdout:?}");
    assert!(!block.ends_with("\n\n\n"), "stdout: {stdout:?}");
    // A blank line precedes the block.
    assert!(
        stdout.contains("\n\n") && !stdout.contains("\n\n\n"),
        "stdout: {stdout:?}"
    );
}

/// benchmark.cpp prints exactly `Unable to open file <path>` (no OS error text) and exits 1.
#[test]
fn bench_with_unreadable_file_reports_the_path_only() {
    let path = "/nonexistent/stockcrab-bench.txt";
    let out = Command::new(env!("CARGO_BIN_EXE_stockcrab"))
        .args(["bench", "16", "1", "8", path, "depth"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr.lines().last(),
        Some(format!("Unable to open file {path}").as_str()),
        "stderr: {stderr:?}"
    );
}
