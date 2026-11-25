#!/bin/bash
# Differential tablebase harness: stockcrab vs the C++ Stockfish reference.
#
# Compares, per position, the bestmove (and ponder) line and the final `info depth` line
# (depth, seldepth, score, nodes, tbhits and PV; nps, time and hashfull are dropped) from
# both engines on random 3-7 man positions. Threads is 1 and every position starts from
# `ucinewgame`, so all of these are deterministic and must be identical. Run it after
# changes to src/syzygy.rs, the tablebase handling in src/search.rs or position/StateInfo
# handling; unit tests and low-man-count runs alone do not exercise 5+ man pawn encodings.
#
# Usage:
#   scripts/syzygy_diff.sh gen <men> <count> <seed>   # write a position file
#   scripts/syzygy_diff.sh run <fenfile> [depth]      # compare both engines
#   scripts/syzygy_diff.sh golden <fenfile>           # emit "FEN|score" from C++
#   scripts/syzygy_diff.sh all                        # generate + run 3-7 men (depth 2),
#                                                     # plus 40 six-man positions at depth 12
#
# Requires:
#   SYZYGY_PATH  directory of tablebase files (see below)
#   the C++ engine built at $STOCKFISH_DIR/src/stockfish (default: ../src/stockfish)
#   the Rust engine built at target/release/stockcrab
# Environment: CPP_ENGINE, RUST_ENGINE, SYZYGY_PATH, SYZYGY_TIMEOUT (extra seconds of
# slack added to the per-engine deadline, default 0), SYZYGY_DIFF_BESTMOVE_ONLY=1
# (compare only the bestmove lines; equal best moves can hide a PV or node divergence).
#
# Exit code: 0 only if every position got an answer from both engines and all
# answers are identical. An engine that exits non-zero, times out, or answers fewer
# positions than requested is a FAILURE (exit 1), never a match; a missing or
# non-executable engine is a hard error (exit 2). Engine stderr is kept in
# $WORK/<name>.err and shown on failure.
#
# Tablebases come from https://tablebase.lichess.ovh/tables/standard/
#   3-4-5-wdl/ + 3-4-5-dtz/   1.0 GB, 290 files -- covers 3, 4 and 5 men
#   6-wdl/ + 6-dtz/           72 GB complete, but individual tables are small:
#                             KPPPPvK 0.4 MB, KNNvKNN 1 MB, KPPPvKP 54 MB
#   7/6v1_pawnful/            KPPPPPvK is 3.7 MB and is the *only* way to reach
#                             five leading pawns, the maximum LEAD_PAWN_IDX is
#                             dimensioned for
#   7/6v1_pawnless/           KQQQQQvK 33 MB, KNNNNNvK 7 MB (+ 285 MB DTZ)
#
# Two pitfalls that make results misleading:
#
#   1. Feed one `isready` after every `go`. Without it, several hundred queued
#      commands drop an occasional `bestmove` under load, and the output shifts
#      by a line -- which reads as hundreds of mismatches.
#   2. Pair positions with results by segmenting the output on `bestmove`, never
#      by zipping two line lists. A position that is already mate emits no
#      `info depth 1` line at all, so a naive zip silently misattributes every
#      later result.
#   3. Depth 2 never reaches the root tablebase PV extension or the in-search probes
#      of positions one capture above the tables: `all` therefore also runs a capped
#      six-man set at depth 12.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
ROOT="$(cd "$HERE/.." && pwd)"
STOCKFISH_DIR="${STOCKFISH_DIR:-$ROOT/..}"
CPP="${CPP_ENGINE:-$STOCKFISH_DIR/src/stockfish}"
RUST="${RUST_ENGINE:-$ROOT/target/release/stockcrab}"
TB="${SYZYGY_PATH:-}"
WORK="${TMPDIR:-/tmp}/syzygy_diff.$$"

die() { echo "ERROR: $*" >&2; exit 2; }
[ -n "$TB" ] || die "set SYZYGY_PATH to a tablebase directory"
[ -d "$TB" ] || die "SYZYGY_PATH is not a directory: $TB"

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

need_engine() { # need_engine <label> <path>
  if [ ! -f "$2" ] || [ ! -x "$2" ]; then die "$1 engine not found/executable at $2"; fi
}

# --- position generator -----------------------------------------------------
# Emits legal positions with `men` pieces. Rejects positions where the side NOT
# to move is in check, which the engines reject as unsupported.
gen() {
  python3 - "$1" "$2" "$3" <<'PY'
import random, sys

def attacks(board, s, p):
    f, r = s % 8, s // 8
    up, t, out = p.isupper(), p.upper(), []
    def ray(df, dr):
        cf, cr = f + df, r + dr
        while 0 <= cf < 8 and 0 <= cr < 8:
            q = cr * 8 + cf; out.append(q)
            if q in board: break
            cf += df; cr += dr
    if t == 'P':
        dr = 1 if up else -1
        for df in (-1, 1):
            cf, cr = f + df, r + dr
            if 0 <= cf < 8 and 0 <= cr < 8: out.append(cr * 8 + cf)
    elif t == 'N':
        for df, dr in ((1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)):
            cf, cr = f + df, r + dr
            if 0 <= cf < 8 and 0 <= cr < 8: out.append(cr * 8 + cf)
    elif t == 'K':
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df or dr:
                    cf, cr = f + df, r + dr
                    if 0 <= cf < 8 and 0 <= cr < 8: out.append(cr * 8 + cf)
    else:
        dirs = []
        if t in 'RQ': dirs += [(1,0),(-1,0),(0,1),(0,-1)]
        if t in 'BQ': dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for d in dirs: ray(*d)
    return out

def in_check(board, white):
    king = next(s for s, p in board.items() if p == ('K' if white else 'k'))
    return any(king in attacks(board, s, p)
               for s, p in board.items() if p.isupper() != white)

def to_fen(board, stm):
    rows = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            c = board.get(r * 8 + f)
            if c is None: empty += 1
            else:
                if empty: row += str(empty); empty = 0
                row += c
        if empty: row += str(empty)
        rows.append(row)
    return "/".join(rows) + f" {stm} - - 0 1"

men, count, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
random.seed(seed)
out, tries = set(), 0
while len(out) < count and tries < 4_000_000:
    tries += 1
    board = {}
    wk, bk = random.randrange(64), random.randrange(64)
    if wk == bk or max(abs(wk % 8 - bk % 8), abs(wk // 8 - bk // 8)) <= 1:
        continue
    board[wk], board[bk] = 'K', 'k'
    ok = True
    for _ in range(men - 2):
        # Pawns are the hard encoding path, so weight them.
        p = random.choice("QRBNPPP")
        s = random.randrange(64)
        if s in board or (p == 'P' and (s < 8 or s >= 56)): ok = False; break
        board[s] = p if random.random() < 0.5 else p.lower()
    if not ok: continue
    stm = random.choice("wb")
    if in_check(board, stm == 'b'): continue
    out.add(to_fen(board, stm))
if len(out) < count:
    sys.exit(f"generator produced only {len(out)} of {count} positions")
for f in sorted(out): print(f)
PY
}

# --- engine driver ----------------------------------------------------------
# drive <name> <engine> <fenfile> <depth> <outfile>
# Feeds every position (after `ucinewgame`, so a divergence cannot cascade into later
# positions through the hash table or the histories) with `isready` after each `go`,
# keeps stdin open until every
# position has produced a `bestmove` (or the deadline passes), then closes it so the
# engine exits by itself. Raw stdout -> outfile, stderr -> $WORK/<name>.err.
# Returns non-zero if the engine exited non-zero or hit the deadline.
drive() {
  local name=$1 engine=$2 fens=$3 depth=$4 out=$5 n rc=0 slack=${SYZYGY_TIMEOUT:-0}
  n=$(grep -cve '^[[:space:]]*$' "$fens" || true)
  : > "$out"
  rm -f "$WORK/$name.rc"
  {
    printf 'setoption name SyzygyPath value %s\nsetoption name Threads value 1\n' "$TB"
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      printf 'ucinewgame\nisready\nposition fen %s\ngo depth %s\nisready\n' "$f" "$depth"
    done < "$fens"
    local waited=0 limit=$((n * 3 + 60 + slack))
    # Stop waiting early if the engine already exited (it writes $name.rc when done).
    while [ ! -e "$WORK/$name.rc" ] && [ "$waited" -lt "$limit" ] \
          && [ "$(grep -ac '^bestmove' "$out" || true)" -lt "$n" ]; do
      sleep 1; waited=$((waited + 1))
    done
  } 2>/dev/null | { tmo $((n * 4 + 120 + slack)) "$engine" >"$out" 2>"$WORK/$name.err" || rc=$?
                    echo "$rc" > "$WORK/$name.rc"; } || true
  # (the feeder may die of SIGPIPE if the engine did; the engine's own status is in $name.rc)
  rc=$(cat "$WORK/$name.rc")
  if [ "$rc" -eq 124 ]; then echo "ERROR: $name engine timed out" >&2; return 1; fi
  if [ "$rc" -ne 0 ]; then echo "ERROR: $name engine exited with status $rc" >&2; return 1; fi
}

# final_info <raw output>: one line per position (segmented on `bestmove`): the last
# `info depth` line before it, without the run-dependent nps/time/hashfull fields.
# A position without any info line gives an empty line, so the pairing stays intact.
final_info() {
  awk '/^info depth / { last = $0 } /^bestmove/ { print last; last = "" }' "$1" \
    | sed -E 's/ (nps|time|hashfull) [0-9]+//g'
}

show_err() { # show_err <name>
  if [ -s "$WORK/$1.err" ]; then echo "--- $1 stderr (last lines) ---" >&2; tail -n 10 "$WORK/$1.err" >&2; fi
}

cmd=${1:-all}
case "$cmd" in
  gen)
    gen "${2:?men}" "${3:?count}" "${4:?seed}"
    ;;

  golden)
    # "FEN|score" for each position, paired by segmenting on bestmove.
    fens=${2:?fenfile}
    need_engine C++ "$CPP"
    [ -s "$fens" ] || die "empty or missing position file: $fens"
    if ! drive cpp "$CPP" "$fens" 1 "$WORK/raw.full"; then show_err cpp; exit 1; fi
    grep -aE '^info depth |^bestmove' "$WORK/raw.full" > "$WORK/raw" || true
    python3 - "$fens" "$WORK/raw" <<'PY'
import sys, re
fens = [l.rstrip("\n") for l in open(sys.argv[1]) if l.strip()]
segs, cur = [], []
for line in open(sys.argv[2]):
    if line.startswith("bestmove"): segs.append(cur); cur = []
    else: cur.append(line.rstrip("\n"))
assert len(segs) == len(fens), f"segments {len(segs)} != positions {len(fens)}"
n = skipped = 0
for fen, seg in zip(fens, segs):
    score = None
    for l in seg:
        m = re.search(r'score (cp|mate) (-?\d+)', l)
        if m: score = f"score {m.group(1)} {m.group(2)}"
    if score: print(f"{fen}|{score}"); n += 1
    else:
        print(f"golden: no score for {fen}", file=sys.stderr); skipped += 1
assert n > 0, "no scored positions"
if skipped: sys.exit(f"{skipped} of {len(fens)} positions had no score")
PY
    ;;

  run)
    fens=${2:?fenfile}; depth=${3:-2}
    need_engine C++ "$CPP"
    need_engine Rust "$RUST"
    [ -s "$fens" ] || die "empty or missing position file: $fens"
    total=$(grep -cve '^[[:space:]]*$' "$fens" || true)
    [ "$total" -gt 0 ] || die "no positions in $fens"

    drive cpp  "$CPP"  "$fens" "$depth" "$WORK/cpp.full"  & pc=$!
    drive rust "$RUST" "$fens" "$depth" "$WORK/rust.full" & pr=$!
    rc_c=0; rc_r=0
    wait "$pc" || rc_c=$?
    wait "$pr" || rc_r=$?
    if [ "$rc_c" -ne 0 ] || [ "$rc_r" -ne 0 ]; then
      echo "  ENGINE FAILURE (cpp rc=$rc_c, rust rc=$rc_r)" >&2; show_err cpp; show_err rust; exit 1
    fi
    grep -aE '^bestmove' "$WORK/cpp.full"  > "$WORK/cpp"  || true
    grep -aE '^bestmove' "$WORK/rust.full" > "$WORK/rust" || true

    ncpp=$(wc -l < "$WORK/cpp"); nrust=$(wc -l < "$WORK/rust")
    printf '  %-28s cpp=%-5s rust=%-5s of %-5s ' "$(basename "$fens")" "$ncpp" "$nrust" "$total"

    if [ "$ncpp" -ne "$total" ] || [ "$nrust" -ne "$total" ]; then
      echo "INCOMPLETE (an engine did not answer every position)"; show_err cpp; show_err rust; exit 1
    fi
    # Full line, so the ponder move is compared too.
    if ! diff -q "$WORK/cpp" "$WORK/rust" >/dev/null; then
      echo "MISMATCH: $({ diff "$WORK/cpp" "$WORK/rust" || true; } | { grep -c '^[<>]' || true; }) lines"
      paste "$fens" "$WORK/cpp" "$WORK/rust" | awk -F'\t' '$2!=$3' | head -10
      exit 1
    fi
    if [ -n "${SYZYGY_DIFF_BESTMOVE_ONLY:-}" ]; then
      echo "IDENTICAL (bestmove + ponder only: SYZYGY_DIFF_BESTMOVE_ONLY is set)"
      exit 0
    fi
    # Equal best moves can hide a different score, PV, node or tbhits count.
    final_info "$WORK/cpp.full"  > "$WORK/cpp.info"
    final_info "$WORK/rust.full" > "$WORK/rust.info"
    if diff -q "$WORK/cpp.info" "$WORK/rust.info" >/dev/null; then
      echo "IDENTICAL (bestmove + ponder + final info line)"
    else
      nbad=$(grep -ve '^[[:space:]]*$' "$fens" | paste - "$WORK/cpp.info" "$WORK/rust.info" \
               | awk -F'\t' '$2!=$3' | wc -l | tr -d ' ')
      echo "INFO MISMATCH $nbad positions"
      grep -ve '^[[:space:]]*$' "$fens" | paste - "$WORK/cpp.info" "$WORK/rust.info" \
        | awk -F'\t' '$2!=$3 { printf "  %s\n    cpp : %s\n    rust: %s\n", $1, $2, $3 }' | head -30
      exit 1
    fi
    ;;

  all)
    rc=0
    for men in 3 4 5 6 7; do
      gen "$men" 120 $((1000 + men)) > "$WORK/f$men.txt" || die "position generation failed for $men men"
      "$0" run "$WORK/f$men.txt" 2 || rc=1
    done
    # Depth 2 never reaches the root tablebase PV extension: a capped six-man set, deeper.
    gen 6 40 2006 > "$WORK/f6-deep.txt" || die "position generation failed for the deep 6-man set"
    "$0" run "$WORK/f6-deep.txt" 12 || rc=1
    exit $rc
    ;;

  *) die "unknown command: $cmd (expected gen|run|golden|all)" ;;
esac
