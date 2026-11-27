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

//! UCI protocol front-end – Rust port of Stockfish's UCIEngine
//!
//! Implements the full UCI protocol for chess engine communication.

use crate::benchmark;
use crate::engine::{Engine, InfoFull, InfoIteration, InfoShort};
use crate::misc::{
    SyncCout, compiler_info, engine_info, engine_version_info, has_large_pages, now,
    split_cpp_whitespace, trim_cpp_whitespace,
};
use crate::movegen::{GenType, generate};
use crate::position::Position;
use crate::search::LimitsType;
use crate::types::{File, Move, MoveType, PieceType, Square, Value};
use core::time::Duration;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Decode one command received over UCI. Modern GUIs send UTF-8. On Windows,
/// older GUIs such as Arena may instead send bytes in the active ANSI code
/// page, so fall back to that encoding when UTF-8 validation fails.
///
/// Never fails: C++ reads the line as raw bytes and carries on whatever they are, so any
/// bytes that cannot be decoded are replaced by U+FFFD for parsing. Error and
/// "Unknown command" messages echo the original bytes instead (see `UciEngine::current_cmd`).
fn decode_uci_command(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(command) => command,
        Err(error) => {
            #[cfg(windows)]
            {
                use windows_sys::Win32::Globalization::{CP_ACP, MultiByteToWideChar};

                let bytes = error.as_bytes();
                if let Ok(byte_len) = i32::try_from(bytes.len()) {
                    // SAFETY: `bytes` is valid for reads of `byte_len` bytes; with a null output buffer and size 0
                    // the API only returns the required length and writes nothing.
                    let wide_len = unsafe {
                        MultiByteToWideChar(
                            CP_ACP,
                            0,
                            bytes.as_ptr().cast(),
                            byte_len,
                            core::ptr::null_mut(),
                            0,
                        )
                    };
                    if wide_len > 0 {
                        let mut wide = vec![0u16; wide_len as usize];
                        // SAFETY: `bytes` is valid for `byte_len` bytes (the length it was converted from) and `wide`
                        // has exactly `wide_len` u16 slots, the capacity the first call reported and this call is given.
                        let converted = unsafe {
                            MultiByteToWideChar(
                                CP_ACP,
                                0,
                                bytes.as_ptr().cast(),
                                byte_len,
                                wide.as_mut_ptr(),
                                wide_len,
                            )
                        };
                        if converted == wide_len {
                            return String::from_utf16_lossy(&wide);
                        }
                    }
                }
            }

            String::from_utf8_lossy(error.as_bytes()).into_owned()
        }
    }
}

/// C-locale `isspace` on a byte (what C++ `istream >> std::string` splits on).
fn is_cpp_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

// Thread-safe output for UCI protocol
static COUT_MUTEX: Mutex<()> = Mutex::new(());

/// Macro for synchronized output
macro_rules! sync_cout {
    ($($arg:tt)*) => {{
        let mut record = format!($($arg)*);
        if !record.ends_with('\n') { record.push('\n'); }
        UciEngine::write_stdout_bytes(record.as_bytes());
    }};
}

macro_rules! sync_cout_endl {
    () => {{
        io::stdout().flush().unwrap();
    }};
}

/// Command line arguments wrapper
pub struct CommandLine {
    pub argc: usize,
    pub argv: Vec<String>,
}

impl CommandLine {
    pub fn new(args: &[String]) -> Self {
        CommandLine {
            argc: args.len(),
            argv: args.to_vec(),
        }
    }
}

// The UCI score type lives in `crate::score` (C++ score.h/score.cpp).
pub use crate::score::Score;

const NUM_WARMUP_POSITIONS: usize = 3;

/// Standard chess starting position FEN (public, as in C++ uci.h).
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Represents the UCI engine with command loop.
pub struct UciEngine {
    engine: Engine,
    cli: CommandLine,
    /// The raw bytes of the command line currently being processed (without the line
    /// terminator, otherwise untrimmed), echoed by critical errors and "Unknown command".
    current_cmd: Vec<u8>,
}

impl UciEngine {
    /// Create a new UCI engine instance.
    pub fn new(args: &[String]) -> Self {
        let binary_path = if args.is_empty() {
            None
        } else {
            Some(args[0].clone().into())
        };
        let mut engine = Engine::new(binary_path);

        // Set options info listener
        engine
            .get_options_mut()
            .add_info_listener(move |str_opt: Option<String>| {
                if let Some(s) = str_opt {
                    Self::print_info_string(&s);
                }
            });

        let mut uci_engine = UciEngine {
            engine,
            cli: CommandLine::new(args),
            current_cmd: Vec::new(),
        };

        // Initialize search update listeners
        uci_engine.init_search_update_listeners();

        uci_engine
    }

    /// Main command loop.
    pub fn loop_(&mut self) {
        let stdin = io::stdin();
        let mut cmd = String::new();
        let mut raw_cmd = Vec::new();

        // Handle command line arguments as one-shot commands
        for i in 1..self.cli.argc {
            cmd.push_str(&self.cli.argv[i]);
            cmd.push(' ');
        }

        loop {
            // Interactive mode if no command line args, otherwise one-shot
            if self.cli.argc <= 1 {
                cmd.clear();
                raw_cmd.clear();
                match stdin.lock().read_until(b'\n', &mut raw_cmd) {
                    Ok(0) => break, // EOF - exit gracefully without stopping engine
                    Ok(_) => {
                        // Like C++ getline: drop the '\n' only (a '\r' stays part of the command).
                        if raw_cmd.last() == Some(&b'\n') {
                            raw_cmd.pop();
                        }
                        cmd = decode_uci_command(raw_cmd.clone());
                    }
                    Err(_) => break, // Error - exit gracefully
                }
                self.current_cmd.clone_from(&raw_cmd);
            } else {
                self.current_cmd = cmd.as_bytes().to_vec();
            }
            crate::misc::with_log_file(|file| {
                let _ = writeln!(file, "<< {}", String::from_utf8_lossy(&self.current_cmd));
            });

            let cmd_trimmed = trim_cpp_whitespace(&cmd);
            if cmd_trimmed.is_empty() && self.cli.argc <= 1 {
                continue;
            }

            let mut tokens = split_cpp_whitespace(cmd_trimmed);
            let token = tokens.next().unwrap_or("");

            match token {
                "quit" | "stop" => {
                    self.engine.stop();
                    if token == "quit" {
                        break;
                    }
                }
                "ponderhit" => self.engine.set_ponderhit(false),
                "uci" => self.uci(),
                "setoption" => self.setoption(cmd_trimmed),
                "go" => {
                    // send info strings after the go command is sent for old GUIs and python-chess
                    Self::print_info_string(&self.engine.numa_config_information_as_string());
                    Self::print_info_string(&self.engine.thread_allocation_information_as_string());
                    self.go(cmd_trimmed);
                }
                "position" => self.position(cmd_trimmed),
                "ucinewgame" => self.engine.search_clear(),
                "isready" => {
                    sync_cout!("readyok");
                    sync_cout_endl!();
                }
                // Custom non-UCI commands, mainly for debugging purposes.
                "flip" => {
                    if let Some(err) = self.engine.flip() {
                        self.terminate_on_critical_error(&err.message_bytes());
                    }
                }
                "bench" => self.bench(cmd_trimmed),
                "speedtest" | "benchmark" => self.benchmark(cmd_trimmed),
                "d" => {
                    sync_cout!("{}", self.engine.visualize());
                    sync_cout_endl!();
                }
                "eval" => self.engine.trace_eval(),
                "compiler" => {
                    // C++ `compiler_info()` starts with "\n" and ends with "\n"; `sync_endl` then adds the
                    // blank line. Ours has no leading newline, so frame it the same way.
                    sync_cout!("\n{}\n", compiler_info());
                    sync_cout_endl!();
                }
                "export_net" => {
                    // export_net takes a single optional output path.
                    let file: Option<std::path::PathBuf> =
                        tokens.next().map(std::path::PathBuf::from);
                    self.apply_idle(|engine| engine.save_network(&file));
                }
                "--help" | "help" | "--license" | "license" => {
                    sync_cout!(
                        "\nStockcrab is a Rust port of Stockfish for playing and analyzing chess.\
                         \nIt is released as free software licensed under the GNU GPLv3 License.\
                         \nStockcrab is normally used with a graphical user interface (GUI) and implements\
                         \nthe Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc.\
                         \nFor any further information, read the README.md and LICENSE files distributed with Stockcrab.\
                         \nUpstream Stockfish: https://github.com/official-stockfish/Stockfish#readme\n\n"
                    );
                    sync_cout_endl!();
                }
                _ if !token.is_empty() && !token.starts_with('#') => {
                    // Echo the raw, untrimmed command bytes like C++ does.
                    let mut out = b"Unknown command: '".to_vec();
                    out.extend_from_slice(&self.current_cmd);
                    out.extend_from_slice(b"'. Type help for more information.\n");
                    Self::write_stdout_bytes(&out);
                }
                _ => {}
            }

            // Command-line arguments are one-shot
            if self.cli.argc > 1 {
                break;
            }
        }
    }

    /// Get the engine's options.
    pub fn engine_options(&self) -> std::sync::RwLockReadGuard<'_, crate::ucioption::OptionsMap> {
        self.engine.get_options()
    }

    /// Current transposition table size, in clusters. Introspection accessor for
    /// `tests/regression_parity.rs` (see `Engine::tt_cluster_count`).
    pub fn tt_cluster_count(&self) -> usize {
        self.engine.tt_cluster_count()
    }

    /// The NUMA topology in C++ `NumaConfig::to_string` form (e.g. `0-3:4-7`).
    /// Introspection accessor for `tests/regression_parity.rs`.
    pub fn numa_config_string(&self) -> String {
        self.engine.get_numa_config_as_string()
    }

    /// Get mutable engine options
    pub fn engine_options_mut(
        &mut self,
    ) -> std::sync::RwLockWriteGuard<'_, crate::ucioption::OptionsMap> {
        self.engine.get_options_mut()
    }

    /// Print an info string in UCI format
    pub fn print_info_string(str: &str) {
        SyncCout::start();
        for line in str.split('\n') {
            if !line.trim().is_empty() {
                sync_cout!("info string {}", line);
            }
        }
        SyncCout::end();
    }

    /// Win rate model parameters based on material count
    fn win_rate_params(pos: &Position) -> (f64, f64) {
        // Count material
        let material = pos.count(PieceType::Pawn)
            + 3 * pos.count(PieceType::Knight)
            + 3 * pos.count(PieceType::Bishop)
            + 5 * pos.count(PieceType::Rook)
            + 9 * pos.count(PieceType::Queen);

        // The fitted model only uses data for material counts in [17, 78], anchored at 58
        let m = (material.clamp(17, 78) as f64) / 58.0;

        // Return a = p_a(material) and b = p_b(material)
        const AS: [f64; 4] = [-142.72052667, 372.35176398, -340.71073572, 415.23490212];
        const BS: [f64; 4] = [5.93832785, 15.61267078, -30.57816876, 69.63866711];

        let a = (((AS[0] * m + AS[1]) * m + AS[2]) * m) + AS[3];
        let b = (((BS[0] * m + BS[1]) * m + BS[2]) * m) + BS[3];

        (a, b)
    }

    /// Win rate model: 1 / (1 + exp((a - eval) / b))
    ///
    /// The win rate model is 1 / (1 + exp((a - eval) / b)), where a = p_a(material) and b = p_b(material).
    /// It fits the LTC fishtest statistics rather accurately.
    fn win_rate_model(v: Value, pos: &Position) -> i32 {
        let (a, b) = Self::win_rate_params(pos);
        // Return win rate in per mille units
        (0.5 + 1000.0 / (1.0 + ((a - v as f64) / b).exp())) as i32
    }

    /// Convert a Value (evaluation units) to centipawns.
    ///
    /// Turns a Value to an integer centipawn number,
    /// without treatment of mate and similar special scores.
    pub fn to_cp(v: Value, pos: &Position) -> i32 {
        // Score can be defined via WDL as:
        // (log(1/L - 1) - log(1/W - 1)) / (log(1/L - 1) + log(1/W - 1))
        // Based on our win_rate_model, this simply yields v / a
        let (a, _b) = Self::win_rate_params(pos);
        (100.0 * v as f64 / a).round() as i32
    }

    /// Format a search score.
    pub fn format_score(s: &Score) -> String {
        const TB_CP: i32 = 20000;

        match s {
            Score::Mate(m) => {
                let plies = m.plies;
                format!("mate {}", if plies > 0 { plies + 1 } else { plies } / 2)
            }
            Score::Tablebase(tb) => {
                format!("cp {}", (if tb.win { TB_CP } else { -TB_CP }) - tb.plies)
            }
            Score::InternalUnits(u) => format!("cp {}", u.value),
        }
    }

    /// Convert a Square to algebraic string, e.g. E2.
    pub fn square(sq: Square) -> String {
        let file = crate::types::file_of(sq);
        let rank = crate::types::rank_of(sq);
        format!(
            "{}{}",
            (b'a' + file as u8) as char,
            (b'1' + rank as u8) as char
        )
    }

    /// Format a move to UCI notation, with optional Chess960 handling.
    pub fn move_str(mv: Move, chess960: bool) -> String {
        if mv == Move::none() {
            return "(none)".into();
        }
        if mv == Move::null() {
            return "0000".into();
        }

        let from = mv.from_sq();
        let mut to = mv.to_sq();

        // Handle castling for standard chess
        if mv.move_type() == MoveType::Castling && !chess960 {
            let file = if to as u8 > from as u8 {
                File::G
            } else {
                File::C
            };
            let rank = crate::types::rank_of(from);
            to = crate::types::make_square(file, rank);
        }

        let mut s = format!("{}{}", Self::square(from), Self::square(to));

        if mv.move_type() == MoveType::Promotion {
            let c = match mv.promotion_type() {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => '?',
            };
            s.push(c);
        }
        s
    }

    /// Win-draw-loss percentages in per-mille: w d l.
    pub fn wdl(v: Value, pos: &Position) -> String {
        let wdl_w = Self::win_rate_model(v, pos);
        let wdl_l = Self::win_rate_model(-v, pos);
        let wdl_d = 1000 - wdl_w - wdl_l;
        format!("{} {} {}", wdl_w, wdl_d, wdl_l)
    }

    /// Lower-case a string.
    pub fn to_lower(str: &str) -> String {
        str.chars().map(|c| c.to_ascii_lowercase()).collect()
    }

    /// Convert UCI move string to Move object
    pub fn to_move(pos: &Position, str: &str) -> Move {
        let str_lower = Self::to_lower(str);

        // Generate all legal moves and find matching one
        let moves = generate(pos, GenType::All);
        for &mv in &moves {
            if str_lower == Self::move_str(mv, pos.is_chess960()) {
                return mv;
            }
        }

        Move::none()
    }

    /// Parse UCI go-limit arguments into `LimitsType`.
    ///
    /// Mirrors C++ `parse_limits`, which extracts values with `istream >> int`: a missing or
    /// non-numeric value fails the stream, and the command is then rejected with
    /// `Err(<token>)` (C++: `terminate_on_critical_error("Invalid argument for '<token>'")`).
    /// Value syntax follows the stream (`5abc` reads 5 and leaves `abc` as the next token).
    pub fn parse_limits(line: &str) -> Result<LimitsType, String> {
        let mut limits = LimitsType::default();
        limits.start_time = now();
        let bytes = line.as_bytes();
        let mut pos = 0usize;

        // `is >> token`
        let next_token = |pos: &mut usize| -> Option<&str> {
            while *pos < bytes.len() && is_cpp_space(bytes[*pos]) {
                *pos += 1;
            }
            let start = *pos;
            while *pos < bytes.len() && !is_cpp_space(bytes[*pos]) {
                *pos += 1;
            }
            (start != *pos).then(|| &line[start..*pos])
        };
        // `is >> value`: `Err(())` is a failed stream.
        let int = |pos: &mut usize, min: i128, max: i128, unsigned: bool| -> Result<i128, ()> {
            crate::misc::stream_extract_int(bytes, pos, min, max, unsigned).map_err(|_| ())
        };
        let (i32min, i32max) = (i128::from(i32::MIN), i128::from(i32::MAX));
        let (i64min, i64max) = (i128::from(i64::MIN), i128::from(i64::MAX));

        while let Some(tok) = next_token(&mut pos) {
            // The stream stays failed after a bad extraction, so only the token that
            // triggered it can be the one reported.
            let ok = match tok {
                "searchmoves" => {
                    // Needs to be the last command on the line: consume the rest as moves
                    while let Some(mv) = next_token(&mut pos) {
                        limits.searchmoves.push(mv.to_ascii_lowercase());
                    }
                    break;
                }
                // `TimePoint` is 64-bit in C++ and in `LimitsType` here.
                "wtime" => int(&mut pos, i64min, i64max, false).map(|v| {
                    limits.time[0] = v as i64;
                }),
                "btime" => int(&mut pos, i64min, i64max, false).map(|v| {
                    limits.time[1] = v as i64;
                }),
                "winc" => int(&mut pos, i64min, i64max, false).map(|v| {
                    limits.inc[0] = v as i64;
                }),
                "binc" => int(&mut pos, i64min, i64max, false).map(|v| {
                    limits.inc[1] = v as i64;
                }),
                "movestogo" => int(&mut pos, i32min, i32max, false).map(|v| {
                    limits.movestogo = v as i32;
                }),
                "depth" => int(&mut pos, i32min, i32max, false).map(|v| {
                    limits.depth = v as i32;
                }),
                "nodes" => int(&mut pos, 0, i128::from(u64::MAX), true).map(|v| {
                    limits.nodes = v as u64;
                }),
                "movetime" => int(&mut pos, i64min, i64max, false).map(|v| {
                    limits.movetime = v as i64;
                }),
                "mate" => int(&mut pos, i32min, i32max, false).map(|v| {
                    limits.mate = v as i32;
                }),
                "perft" => int(&mut pos, i32min, i32max, false).map(|v| {
                    limits.perft = v as i32;
                }),
                "infinite" => {
                    limits.infinite = 1;
                    Ok(())
                }
                "ponder" => {
                    limits.ponder_mode = true;
                    Ok(())
                }
                _ => Ok(()),
            };
            if ok.is_err() {
                return Err(tok.to_string());
            }
        }

        // Sanitize negative time values, zero for stm means no TM
        let [wtime, btime] = &mut limits.time;
        for t in [wtime, btime, &mut limits.movetime] {
            if *t < 0 {
                *t = 1;
            }
        }

        Ok(limits)
    }

    /// `parse_limits`, terminating the process on an invalid argument like C++.
    fn parse_limits_or_exit(&self, line: &str) -> LimitsType {
        Self::parse_limits(line).unwrap_or_else(|token| {
            self.terminate_on_critical_error(format!("Invalid argument for '{token}'").as_bytes())
        })
    }

    // UCI command handlers

    /// Handle 'uci' command - print engine ID and options.
    fn uci(&self) {
        // C++ is two statements:
        //     sync_cout << "id name " << engine_info(true) << "\n" << options << sync_endl;
        //     sync_cout << "uciok" << sync_endl;
        // The first `sync_endl` terminates the options block, so `uciok` is on a line of its own.
        sync_cout!("id name {}", engine_info(true));
        sync_cout!("{}", self.engine.get_options());
        sync_cout_endl!();
        sync_cout!("uciok");
        sync_cout_endl!();
    }

    /// Handle 'position' command - set position from FEN or startpos + moves.
    fn position(&mut self, line: &str) {
        let mut tokens = split_cpp_whitespace(line);
        tokens.next(); // skip "position"

        let mut moves = Vec::new();

        match tokens.next() {
            Some("startpos") => {
                // Skip to moves if present
                for token in tokens.by_ref() {
                    if token == "moves" {
                        break;
                    }
                }
                // Collect moves
                for move_str in tokens {
                    moves.push(move_str.to_string());
                }
                if let Some(err) = self.engine.set_position(START_FEN, &moves) {
                    self.terminate_on_critical_error(&err.message_bytes());
                }
            }
            Some("fen") => {
                // Collect FEN parts until "moves" or end
                let mut fen_parts = Vec::new();
                for token in tokens.by_ref() {
                    if token == "moves" {
                        break;
                    }
                    fen_parts.push(token);
                }
                // Each token is followed by a space, matching C++
                // `fen += token + " "`. The trailing space is significant:
                // `Position::set` requires a whitespace character after the side
                // to move, so joining without it would reject FENs that C++
                // accepts (e.g. a board plus a bare "w").
                let fen: String = fen_parts.iter().map(|t| format!("{} ", t)).collect();

                // Collect moves
                for move_str in tokens {
                    moves.push(move_str.to_string());
                }

                // An empty FEN is passed on too: C++ reports "Unexpected end of stream".
                if let Some(err) = self.engine.set_position(&fen, &moves) {
                    self.terminate_on_critical_error(&err.message_bytes());
                }
            }
            _ => (),
        }
    }

    /// Handle 'go' command - start search with given limits.
    fn go(&mut self, line: &str) {
        let limits = self.parse_limits_or_exit(line);

        // C++ `if (limits.perft)`: any nonzero value, negative included, runs perft.
        if limits.perft != 0 {
            self.perft(&limits);
        } else {
            self.engine.go(limits);
        }
    }

    /// Run perft (performance test)
    fn perft(&mut self, limits: &LimitsType) -> u64 {
        let fen = self.engine.fen();
        let chess960 = self
            .engine
            .get_options()
            .get("UCI_Chess960")
            .map(|opt| opt.is_true())
            .unwrap_or(false);
        // Engine::perft returns either a node count or a position setup error; the error is
        // reported and the process exits, as in C++.
        let nodes = match self.engine.perft(&fen, limits.perft, chess960) {
            Ok(n) => n,
            Err(err) => self.terminate_on_critical_error(&err.message_bytes()),
        };
        sync_cout!("\nNodes searched: {}\n\n", nodes);
        sync_cout_endl!();
        nodes
    }

    /// Handle 'setoption' command - set UCI option value.
    ///
    /// Public so that `tests/regression_parity.rs` can drive the real dispatch
    /// path (option store + on-change side effects) without going through stdin.
    pub fn setoption(&mut self, line: &str) {
        let rest = line.trim_start();
        let rest = rest.strip_prefix("setoption").unwrap_or(rest).trim_start();
        for message in self.apply_idle(|engine| engine.set_option(rest)) {
            Self::print_info_string(&message);
        }
    }

    /// Apply a mutating command only when idle: stop any running search and wait for it to
    /// finish first, so a non-conforming GUI cannot change the engine under a search.
    fn apply_idle<R>(&mut self, mutate: impl FnOnce(&mut Engine) -> R) -> R {
        self.engine.stop();
        self.engine.wait_for_search_finished();
        mutate(&mut self.engine)
    }

    /// Handle 'bench' command - run benchmark.
    fn bench(&mut self, line: &str) {
        let args = line.strip_prefix("bench").unwrap_or("").trim();
        let current_fen = self.engine.fen();
        let commands = benchmark::setup_bench(&current_fen, args);

        let mut start = now();
        let mut nodes = 0u64;
        let mut cnt = 1;

        let num_positions = commands
            .iter()
            .filter(|cmd| cmd.starts_with("go ") || cmd.starts_with("eval"))
            .count();

        for cmd in commands {
            let mut cmd_tokens = split_cpp_whitespace(&cmd);
            match cmd_tokens.next() {
                Some("position") => self.position(&cmd),
                Some("setoption") => self.setoption(&cmd),
                Some("ucinewgame") => {
                    self.engine.search_clear(); // search_clear may take a while
                    start = now(); // C++ restarts the clock here (uci.cpp): setup is not timed
                }
                Some("go") => {
                    eprintln!(
                        "\nPosition: {}/{} ({})",
                        cnt,
                        num_positions,
                        self.engine.fen()
                    );
                    cnt += 1;

                    // `bench <tt> <threads> <n> <fens> perft` reaches here as `go perft n`;
                    // it is routed to perft because `Engine::go` requires `limits.perft == 0`.
                    // C++ splits the two cases the same way (uci.cpp).
                    let limits = self.parse_limits_or_exit(&cmd);
                    if limits.perft != 0 {
                        nodes += self.perft(&limits);
                    } else {
                        self.engine.go(limits);
                        self.engine.wait_for_search_finished();
                        nodes += self.engine.nodes_searched();
                    }
                }
                Some("eval") => {
                    eprintln!(
                        "\nPosition: {}/{} ({})",
                        cnt,
                        num_positions,
                        self.engine.fen()
                    );
                    cnt += 1;
                    self.engine.trace_eval();
                }
                _ => {}
            }
        }

        let elapsed = now() - start + 1; // Ensure positivity to avoid a 'divide by zero' (C++ uci.cpp)
        let nps = nodes * 1000 / elapsed as u64;

        crate::misc::dbg_print();

        // C++ writes the summary to stderr (`std::cerr`), after a blank line.
        eprintln!("\n===========================");
        eprintln!("Total time (ms) : {}", elapsed);
        eprintln!("Nodes searched  : {}", nodes);
        eprintln!("Nodes/second    : {}", nps);
    }

    /// Handle 'benchmark' command - run extended benchmark.
    fn benchmark(&mut self, line: &str) {
        let args = if line.starts_with("speedtest") {
            line.strip_prefix("speedtest")
        } else {
            line.strip_prefix("benchmark")
        }
        .unwrap_or("")
        .trim();

        let setup = benchmark::setup_benchmark(args);

        // Silence all search callbacks; the timing/node listeners are installed
        // only for the measured run, after the warmup.
        self.engine.set_on_update_full(|_info: InfoFull| {});
        self.engine.set_on_iter(|_info: InfoIteration| {});
        self.engine.set_on_update_no_moves(|_info: InfoShort| {});
        self.engine.set_on_bestmove(|_bm: &str, _ponder: &str| {});
        self.engine.set_on_verify_network(|_s: &str| {});

        // Set options once
        self.setoption(&format!("setoption name Threads value {}", setup.threads));
        self.setoption(&format!("setoption name Hash value {}", setup.tt_size));
        self.setoption("setoption name UCI_Chess960 value false");

        let num_go_commands = setup
            .commands
            .iter()
            .filter(|cmd| cmd.starts_with("go "))
            .count();

        // Warmup phase
        let mut cnt = 1;

        for cmd in &setup.commands {
            let mut cmd_tokens = split_cpp_whitespace(cmd);
            match cmd_tokens.next() {
                Some("go") => {
                    // One new line is produced by the search, so omit it here
                    eprint!("\rWarmup position {}/{}", cnt, NUM_WARMUP_POSITIONS);

                    let limits = self.parse_limits_or_exit(cmd);

                    self.engine.go(limits);
                    self.engine.wait_for_search_finished();

                    cnt += 1;
                    if cnt > NUM_WARMUP_POSITIONS {
                        break;
                    }
                }
                Some("position") => self.position(cmd),
                Some("ucinewgame") => self.engine.search_clear(),
                _ => {}
            }
        }

        eprintln!();

        // Main benchmark run
        cnt = 1;

        let mut num_hashfull_readings = 0;
        let hashfull_ages = [0, 999]; // Normal and touched hash
        let mut total_hashfull = [0i32; 2];
        let mut max_hashfull = [0i32; 2];

        self.engine.search_clear();

        // Time only the searches themselves: the clock is started by the engine's
        // on-start callback and stopped when the bestmove is reported, so neither
        // the go/wait plumbing nor `search_clear` pollutes the measurement.
        let elapsed = Arc::new(Mutex::new(Instant::now()));
        let total_time = Arc::new(Mutex::new(Duration::ZERO));
        let nodes = Arc::new(Mutex::new(0u64));
        let nodes_searched = Arc::new(Mutex::new(0u64));

        {
            let nodes_searched = Arc::clone(&nodes_searched);
            self.engine.set_on_update_full(move |info: InfoFull| {
                *nodes_searched.lock().unwrap() = info.nodes;
            });
        }

        {
            let elapsed = Arc::clone(&elapsed);
            let nodes_searched = Arc::clone(&nodes_searched);
            self.engine.set_on_start(move || {
                *elapsed.lock().unwrap() = Instant::now();
                *nodes_searched.lock().unwrap() = 0;
            });
        }

        {
            let elapsed = Arc::clone(&elapsed);
            let total_time = Arc::clone(&total_time);
            let nodes = Arc::clone(&nodes);
            let nodes_searched = Arc::clone(&nodes_searched);
            self.engine
                .set_on_bestmove(move |_bm: &str, _ponder: &str| {
                    let start = *elapsed.lock().unwrap();
                    *total_time.lock().unwrap() += Instant::now() - start;
                    *nodes.lock().unwrap() += *nodes_searched.lock().unwrap();
                });
        }

        for cmd in &setup.commands {
            let mut cmd_tokens = split_cpp_whitespace(cmd);
            match cmd_tokens.next() {
                Some("go") => {
                    // One new line is produced by the search, so omit it here
                    eprint!("\rPosition {}/{}", cnt, num_go_commands);

                    let limits = self.parse_limits_or_exit(cmd);

                    self.engine.go(limits);
                    self.engine.wait_for_search_finished();

                    // Update hashfull readings
                    num_hashfull_readings += 1;
                    for i in 0..hashfull_ages.len() {
                        let hashfull = self.engine.get_hashfull(hashfull_ages[i]);
                        max_hashfull[i] = max_hashfull[i].max(hashfull);
                        total_hashfull[i] += hashfull;
                    }

                    cnt += 1;
                }
                Some("position") => self.position(cmd),
                Some("ucinewgame") => self.engine.search_clear(),
                _ => {}
            }
        }

        let nodes = *nodes.lock().unwrap();
        // Ensure positivity to avoid a 'divide by zero'
        let total_time_ms = (total_time.lock().unwrap().as_millis() as u64).max(1);

        eprintln!();

        let thread_binding = self.engine.thread_binding_information_as_string();
        let thread_binding = if thread_binding.is_empty() {
            "none".to_string()
        } else {
            thread_binding
        };

        // Output results
        eprintln!("===========================");
        eprintln!("Version                    : {}", engine_version_info());
        eprint!("{}", compiler_info());
        eprintln!(
            "Large pages                : {}",
            if has_large_pages() { "yes" } else { "no" }
        );
        eprintln!(
            "User invocation            : speedtest {}",
            setup.original_invocation
        );
        eprintln!(
            "Filled invocation          : speedtest {}",
            setup.filled_invocation
        );
        eprintln!(
            "Available processors       : {}",
            self.engine.get_numa_config_as_string()
        );
        eprintln!("Thread count               : {}", setup.threads);
        eprintln!("Thread binding             : {}", thread_binding);
        eprintln!("TT size [MiB]              : {}", setup.tt_size);
        eprintln!("Hash max, avg [per mille]  : ");
        eprintln!(
            "    single search          : {}, {}",
            max_hashfull[0],
            if num_hashfull_readings > 0 {
                total_hashfull[0] / num_hashfull_readings
            } else {
                0
            }
        );
        eprintln!(
            "    single game            : {}, {}",
            max_hashfull[1],
            if num_hashfull_readings > 0 {
                total_hashfull[1] / num_hashfull_readings
            } else {
                0
            }
        );
        eprintln!("Total nodes searched       : {}", nodes);
        eprintln!(
            "Total search time [s]      : {:.3}",
            total_time_ms as f64 / 1000.0
        );
        eprintln!(
            "Nodes/second               : {}",
            1000 * nodes / total_time_ms
        );

        crate::misc::dbg_print();

        // Reset callbacks
        self.init_search_update_listeners();
    }

    // Search update callbacks

    /// Initialize search update listeners
    fn init_search_update_listeners(&mut self) {
        // Read the option at emit time so runtime changes to UCI_ShowWDL take effect. C++
        // reads `options["UCI_ShowWDL"]` inside the lambda for the same reason (uci.cpp).
        let options = self.engine.options_handle();

        self.engine.set_on_iter(|info: InfoIteration| {
            Self::on_iter(&info);
        });

        self.engine.set_on_update_no_moves(|info: InfoShort| {
            Self::on_update_no_moves(&info);
        });

        self.engine.set_on_update_full(move |info: InfoFull| {
            let show_wdl = options
                .read()
                .ok()
                .and_then(|o| o.get("UCI_ShowWDL").map(|opt| opt.is_true()))
                .unwrap_or(false);
            Self::on_update_full(&info, show_wdl);
        });

        self.engine.set_on_start(|| {});

        self.engine.set_on_bestmove(|bestmove: &str, ponder: &str| {
            Self::on_bestmove(bestmove, ponder);
        });

        self.engine.set_on_verify_network(|s: &str| {
            Self::print_info_string(s);
        });
    }

    /// Handle search update with no legal moves
    fn on_update_no_moves(info: &InfoShort) {
        let score = info.score;

        sync_cout!(
            "info depth {} score {}",
            info.depth,
            Self::format_score(&score)
        );
        sync_cout_endl!();
    }

    /// Handle full search update
    fn on_update_full(info: &InfoFull, show_wdl: bool) {
        let bound = if info.bound.is_empty() {
            String::new()
        } else {
            format!(" {}", info.bound)
        };
        let wdl = if show_wdl {
            format!(" wdl {}", info.wdl)
        } else {
            String::new()
        };
        sync_cout!(
            "info depth {} seldepth {} multipv {} score {}{}{} nodes {} nps {} hashfull {} tbhits {} time {} pv {}",
            info.depth,
            info.sel_depth,
            info.multi_pv,
            Self::format_score(&info.score),
            bound,
            wdl,
            info.nodes,
            info.nps,
            info.hashfull,
            info.tb_hits,
            info.time_ms,
            info.pv,
        );
    }

    /// Handle iteration info
    fn on_iter(info: &InfoIteration) {
        sync_cout!(
            "info depth {} currmove {} currmovenumber {}",
            info.depth,
            info.currmove,
            info.currmovenumber
        );
    }

    /// Handle best move output
    fn on_bestmove(bestmove: &str, ponder: &str) {
        if ponder.is_empty() {
            sync_cout!("bestmove {}", bestmove);
        } else {
            sync_cout!("bestmove {} ponder {}", bestmove, ponder);
        }
    }

    /// Report a critical error for the current command and terminate the process.
    /// Mirrors C++ `UCIEngine::terminate_on_critical_error` (marked `[[noreturn]]`).
    fn terminate_on_critical_error(&self, message: &[u8]) -> ! {
        let mut out = b"info string CRITICAL ERROR: Command `".to_vec();
        out.extend_from_slice(&self.current_cmd);
        out.extend_from_slice(b"` failed. Reason: ");
        out.extend_from_slice(message);
        out.extend_from_slice(b"\n\n");
        Self::write_stdout_bytes(&out);
        std::process::exit(1);
    }

    /// Write raw bytes to stdout under the output lock and flush (C++ `sync_cout << ... << sync_endl`).
    pub(crate) fn write_stdout_bytes(bytes: &[u8]) {
        let _lock = COUT_MUTEX.lock().unwrap();
        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(bytes);
        let _ = stdout.flush();
        crate::misc::with_log_file(|file| {
            for line in String::from_utf8_lossy(bytes).lines() {
                let _ = writeln!(file, ">> {line}");
            }
        });
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::types::{Move, PieceType, Square};

    #[test]
    fn test_to_lower() {
        assert_eq!(UciEngine::to_lower("TeStHi"), "testhi");
    }

    #[test]
    fn test_decode_uci_command_utf8() {
        assert_eq!(
            decode_uci_command(
                "setoption name SyzygyPath value /tmp/é\n"
                    .as_bytes()
                    .to_vec()
            ),
            "setoption name SyzygyPath value /tmp/é\n".to_string()
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn test_decode_uci_command_is_lossy_off_windows() {
        assert_eq!(decode_uci_command(vec![b'a', 0xff, b'\n']), "a\u{fffd}\n");
    }

    #[test]
    fn test_square() {
        assert_eq!(UciEngine::square(Square::SQ_A1), "a1");
        assert_eq!(UciEngine::square(Square::SQ_E2), "e2");
    }

    #[test]
    fn test_move_str_quiet() {
        let mv = Move::make_normal(Square::SQ_E2, Square::SQ_E4);
        assert_eq!(UciEngine::move_str(mv, false), "e2e4");
    }

    #[test]
    fn test_move_str_promotion() {
        let mv = Move::make_promotion(Square::SQ_E7, Square::SQ_E8, PieceType::Queen);
        assert_eq!(UciEngine::move_str(mv, false), "e7e8q");
    }

    #[test]
    fn test_move_str_castling() {
        let mv = Move::make_castling(Square::SQ_E1, Square::SQ_H1);
        assert_eq!(UciEngine::move_str(mv, false), "e1g1"); // King-side castling

        let mv = Move::make_castling(Square::SQ_E1, Square::SQ_A1);
        assert_eq!(UciEngine::move_str(mv, false), "e1c1"); // Queen-side castling
    }

    #[test]
    fn test_wdl() {
        let pos = Position::default();
        let wdl = UciEngine::wdl(0, &pos);
        // Should return values that sum to 1000
        let parts: Vec<i32> = wdl
            .split_whitespace()
            .map(|s| s.parse::<i32>().expect("WDL field"))
            .collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0] + parts[1] + parts[2], 1000);
        // At v=0 win and loss are symmetric.
        assert_eq!(parts[0], parts[2]);
    }

    #[test]
    fn test_parse_limits() {
        let lim = UciEngine::parse_limits(
            "wtime 3000 btime 2000 winc 5 binc 7 movestogo 40 depth 10 nodes 1234 movetime 66 mate 5 perft 3 infinite ponder searchmoves e2e4 g1f3",
        )
        .unwrap();
        assert_eq!(lim.time[0], 3000);
        assert_eq!(lim.time[1], 2000);
        assert_eq!(lim.inc[0], 5);
        assert_eq!(lim.inc[1], 7);
        assert_eq!(lim.movestogo, 40);
        assert_eq!(lim.depth, 10);
        assert_eq!(lim.nodes, 1234);
        assert_eq!(lim.movetime, 66);
        assert_eq!(lim.mate, 5);
        assert_eq!(lim.perft, 3);
        assert_eq!(lim.infinite, 1);
        assert!(lim.ponder_mode);
        assert_eq!(lim.searchmoves, vec!["e2e4", "g1f3"]);
    }

    /// Negative remaining or move times become 1 ms (zero would mean no time management);
    /// increments are kept as given, also after `searchmoves` ends the parse.
    #[test]
    fn test_parse_limits_sanitizes_negative_times() {
        let lim = UciEngine::parse_limits(
            "wtime -50 btime 0 winc -5 binc -7 movetime -9223372036854775808 searchmoves e2e4",
        )
        .unwrap();
        assert_eq!(lim.time, [1, 0]);
        assert_eq!(lim.inc, [-5, -7]);
        assert_eq!(lim.movetime, 1);
        let lim = UciEngine::parse_limits("btime -1 wtime 10").unwrap();
        assert_eq!(lim.time, [10, 1]);
        assert_eq!(lim.movetime, 0);
    }

    #[test]
    fn test_parse_limits_invalid_arguments() {
        // A missing or non-numeric value names the offending token (C++: stream failbit).
        assert_eq!(
            UciEngine::parse_limits("go depth 3 mate abc")
                .err()
                .as_deref(),
            Some("mate")
        );
        assert_eq!(
            UciEngine::parse_limits("go depth 1 nodes").err().as_deref(),
            Some("nodes")
        );
        assert_eq!(
            UciEngine::parse_limits("go depth 2 wtime e2e4")
                .err()
                .as_deref(),
            Some("wtime")
        );
        assert_eq!(
            UciEngine::parse_limits("go depth 1 depth x")
                .err()
                .as_deref(),
            Some("depth")
        );
        assert_eq!(
            UciEngine::parse_limits("go depth 99999999999")
                .err()
                .as_deref(),
            Some("depth")
        );
        // Stream quirks: trailing garbage is left as the next token; a signed target takes a
        // minus sign; an unsigned one wraps like strtoull.
        let lim = UciEngine::parse_limits("go depth 5abc movestogo +7 mate -2 nodes -1").unwrap();
        assert_eq!((lim.depth, lim.movestogo, lim.mate), (5, 7, -2));
        assert_eq!(lim.nodes, u64::MAX);
        // Unknown tokens are ignored.
        assert!(UciEngine::parse_limits("go foo bar infinite").is_ok());
    }
}
