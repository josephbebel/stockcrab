#!/usr/bin/env python3
"""Randomised differential test: stockcrab vs C++ Stockfish.

Plays random legal games (move lists come from the C++ engine's `go perft 1`), samples
positions along the way, and compares on each: the position state (`d`: FEN, key,
checkers), perft(2), the static `eval` output and a fixed-depth search (best move, score
and node count). Node-for-node identical search is the port's central claim, so any
difference is a bug in one of the two engines.

Every position is sent as `position fen <start> moves <full move history>` (never just
the final FEN) so repetition and 50-move state are exercised in both engines. Games
sometimes shuffle pieces back and forth to create repetitions, and some start FENs carry
a high halfmove clock to reach the 50-move rule.

With --chess960 each game starts from a random Chess960 layout (UCI_Chess960 is set),
written as a Shredder-FEN (file letters) or X-FEN (KQkq) castling field; a share of the
layouts is chosen so that the king and/or a rook already stands on its castling
destination square.

Exit status: 0 = all positions agree; 1 = at least one mismatch; 2 = an engine failed
(exited non-zero, timed out, closed its output early or answered nonsensically). Failed
executions are never treated as agreement. Engine stderr goes to a log file per engine
(--log-dir), whose tail is shown on failure.

Usage:
  scripts/fuzz_diff.py [--positions N] [--seed S] [--depth D] [--chess960] [--timeout SEC]

Environment (same as scripts/parity.sh):
  CPP_ENGINE   C++ engine  (default: $STOCKFISH_DIR/src/stockfish, STOCKFISH_DIR=..)
  RUST_ENGINE  Rust engine (default: target/release/stockcrab)
Run from a directory where both engines find the NNUE network (e.g. $STOCKFISH_DIR/src).
"""
import argparse
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.environ.get("STOCKFISH_DIR", os.path.join(ROOT, ".."))
CPP = os.environ.get("CPP_ENGINE", os.path.join(SF, "src", "stockfish"))
RUST = os.environ.get("RUST_ENGINE", os.path.join(ROOT, "target", "release", "stockcrab"))
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class EngineError(Exception):
    """An engine misbehaved (not a parity mismatch)."""


class Engine:
    def __init__(self, name, path, chess960, timeout, log_dir):
        self.name, self.timeout = name, timeout
        self.log_path = os.path.join(log_dir, f"{name}.stderr.log")
        self.err = open(self.log_path, "w")
        env = dict(os.environ, RUST_MIN_STACK="50000000")
        try:
            self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self.err, text=True, env=env, bufsize=1)
        except OSError as e:
            raise EngineError(f"{name}: cannot start {path}: {e}")
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.until("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        if chess960:
            self.send("setoption name UCI_Chess960 value true")
        self.send("isready")
        self.until("readyok")

    def _reader(self):
        try:
            for line in self.p.stdout:
                self.q.put(line.rstrip("\n"))
        except (ValueError, OSError):
            pass
        self.q.put(None)  # EOF marker

    def send(self, cmd):
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise EngineError(f"{self.name}: engine closed its input while sending '{cmd}' "
                              f"(exit status {self.p.poll()})")

    def until(self, prefix, timeout=None):
        """Collect lines until one starts with `prefix`; the deadline is per command."""
        deadline = time.monotonic() + (timeout or self.timeout)
        out = []
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.p.kill()
                raise EngineError(f"{self.name}: timed out waiting for '{prefix}'")
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                try:
                    rc = self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    rc = "still running"
                raise EngineError(f"{self.name}: engine ended before '{prefix}' (exit status {rc})")
            out.append(line)
            if line.startswith(prefix):
                return out

    def ask(self, cmds, prefix):
        for c in cmds:
            self.send(c)
        return self.until(prefix)

    def stderr_tail(self, n=15):
        try:
            self.err.flush()
        except ValueError:
            pass  # already closed
        try:
            with open(self.log_path) as f:
                return "".join(f.readlines()[-n:])
        except (OSError, ValueError):
            return ""

    def close(self):
        """Quit and require a clean exit status."""
        try:
            self.send("quit")
            rc = self.p.wait(timeout=10)
        except (EngineError, subprocess.TimeoutExpired):
            self.p.kill()
            raise EngineError(f"{self.name}: did not exit after 'quit'")
        finally:
            self.err.close()
        if rc != 0:
            raise EngineError(f"{self.name}: exited with status {rc}")


def pos_cmd(start, moves):
    return f"position fen {start}" + (" moves " + " ".join(moves) if moves else "")


def legal_moves(eng, start, moves):
    lines = eng.ask([pos_cmd(start, moves), "go perft 1"], "Nodes searched")
    return [m.group(1) for l in lines if (m := re.match(r"^([a-h][1-8][a-h][1-8][qrbn]?): 1", l))]


def state(eng, start, moves):
    lines = eng.ask([pos_cmd(start, moves), "d"], "Checkers")
    got = [l for l in lines if l.startswith(("Fen:", "Key:", "Checkers:"))]
    if not any(l.startswith("Fen:") for l in got):
        raise EngineError(f"{eng.name}: no Fen line from `d`")
    return got


def perft2(eng, start, moves):
    lines = eng.ask([pos_cmd(start, moves), "go perft 2"], "Nodes searched")
    return lines[-1]


def static_eval(eng, start, moves):
    lines = eng.ask([pos_cmd(start, moves), "eval"], "Final evaluation")
    got = [l for l in lines if l.startswith(("Final evaluation", "NNUE evaluation"))]
    if not got:
        raise EngineError(f"{eng.name}: no evaluation line")
    return got


def search(eng, start, moves, depth):
    lines = eng.ask(["ucinewgame", pos_cmd(start, moves), f"go depth {depth}"], "bestmove")
    last = [l for l in lines if l.startswith("info depth") and " score " in l]
    parts = lines[-1].split()
    best = parts[1] if len(parts) > 1 else None
    if not last or best in (None, "(none)", "0000"):
        raise EngineError(f"{eng.name}: search from a position with legal moves produced "
                          f"no info line or no bestmove ({lines[-3:]})")
    sc = re.search(r"score (cp|mate) (-?\d+)", last[-1])
    nodes = re.search(r" nodes (\d+)", last[-1])
    if not sc or not nodes:
        raise EngineError(f"{eng.name}: unparsable search info line: {last[-1]}")
    return (best, sc.group(0), nodes.group(1))


def chess960_start(rng, targeted):
    """A random valid Chess960 start FEN (bishops on opposite colours, king between the
    rooks). `targeted` picks only layouts where the king and/or a rook already stands on
    its castling destination (king on c/g, rooks on d/f)."""
    while True:
        r = [None] * 8
        r[rng.choice([0, 2, 4, 6])] = "b"
        r[rng.choice([1, 3, 5, 7])] = "b"
        free = [i for i in range(8) if r[i] is None]
        r[free.pop(rng.randrange(len(free)))] = "q"
        for _ in range(2):
            r[free.pop(rng.randrange(len(free)))] = "n"
        r[free[0]], r[free[1]], r[free[2]] = "r", "k", "r"
        kf = r.index("k")
        rooks = [i for i, p in enumerate(r) if p == "r"]
        if targeted and not (kf in (2, 6) or rooks[0] == 3 or rooks[1] == 5):
            continue
        break
    back = "".join(r)
    if rng.random() < 0.5:  # Shredder-FEN: castling rights as the rooks' file letters
        rights = "ABCDEFGH"[rooks[1]] + "ABCDEFGH"[rooks[0]]
        rights += rights.lower()
    else:                    # X-FEN: unambiguous here (exactly two rooks, king between)
        rights = "KQkq"
    return f"{back}/pppppppp/8/8/8/8/PPPPPPPP/{back.upper()} w {rights} - 0 1"


def random_game(cpp, rng, chess960, targeted):
    """Returns (start_fen, moves) ending in a position that has legal moves."""
    if chess960:
        start = chess960_start(rng, targeted)
    elif rng.random() < 0.15:
        # high halfmove clock so the 50-move rule is reachable within the game
        start = START.replace("0 1", f"{rng.randrange(60, 98)} 1")
    else:
        start = START
    moves = []
    for _ in range(rng.randrange(8, 90) if rng.random() < 0.85 else rng.randrange(90, 220)):
        legal = legal_moves(cpp, start, moves)
        if not legal:
            break
        choice = None
        if moves and rng.random() < 0.3:
            # try the reverse squares of the previous move (promotions and castling are not truly
            # undone): shuffles create repetitions
            back = moves[-1][2:4] + moves[-1][0:2]
            if back in legal:
                choice = back
        moves.append(choice or rng.choice(legal))
    return start, moves


def run(a, log_dir):
    for label, path in (("C++", CPP), ("Rust", RUST)):
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            print(f"ERROR: {label} engine not found/executable: {path}", file=sys.stderr)
            return 2
    rng = random.Random(a.seed)
    cpp = rust = None
    mismatches = checked = attempts = 0
    try:
        cpp = Engine("cpp", CPP, a.chess960, a.timeout, log_dir)
        rust = Engine("rust", RUST, a.chess960, a.timeout, log_dir)
        while checked < a.positions:
            attempts += 1
            if attempts > a.positions * 50 + 50:
                raise EngineError("could not generate enough positions with legal moves")
            start, moves = random_game(cpp, rng, a.chess960, rng.random() < 0.4)
            if not legal_moves(cpp, start, moves):
                continue  # mate/stalemate: nothing to search
            checked += 1
            where = f"{pos_cmd(start, moves)}"
            for what, fn in (("state", state), ("perft2", perft2), ("eval", static_eval),
                             ("search", lambda e, s, m: search(e, s, m, a.depth))):
                c, r = fn(cpp, start, moves), fn(rust, start, moves)
                if c != r:
                    mismatches += 1
                    print(f"MISMATCH {what}: {where}\n  cpp : {c}\n  rust: {r}")
            if checked % 25 == 0:
                print(f"  {checked}/{a.positions} positions, {mismatches} mismatches", flush=True)
        for e in (cpp, rust):
            e.close()
        cpp = rust = None
    except EngineError as e:
        print(f"ENGINE FAILURE: {e}", file=sys.stderr)
        for eng in (cpp, rust):
            if eng is not None:
                tail = eng.stderr_tail()
                if tail:
                    print(f"--- {eng.name} stderr (last lines; full log {eng.log_path}) ---\n{tail}",
                          file=sys.stderr)
        return 2
    finally:
        for eng in (cpp, rust):
            if eng is not None and eng.p.poll() is None:
                eng.p.kill()
    if checked != a.positions:
        print(f"ERROR: compared {checked} of {a.positions} positions", file=sys.stderr)
        return 2
    chess = " (Chess960)" if a.chess960 else ""
    print(f"{checked} positions{chess}, {mismatches} mismatches")
    if mismatches:
        print(f"engine stderr logs: {log_dir}")
    return 1 if mismatches else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--chess960", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="deadline in seconds for each engine command (default 120)")
    ap.add_argument("--log-dir", default=None, help="directory for engine stderr logs")
    a = ap.parse_args()
    if a.positions < 1:
        sys.exit("--positions must be at least 1")
    log_dir = a.log_dir or tempfile.mkdtemp(prefix="fuzz_diff.")
    os.makedirs(log_dir, exist_ok=True)
    sys.exit(run(a, log_dir))


if __name__ == "__main__":
    main()
