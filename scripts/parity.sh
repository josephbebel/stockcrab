#!/bin/bash
# Differential parity harness: stockcrab (Rust) vs upstream Stockfish (C++).
#
# Run after every upstream sync and functional change; a passing bench node count
# alone does not establish parity.
#
# Prerequisites:
#   1. Build the C++ reference:  cd ../src && make -j8 build ARCH=<your-arch>
#   2. Build the Rust engine:    cargo build --release --bin stockcrab
#   3. Both engines must use the SAME network file (symlink it into ../src/).
#
# Environment: STOCKFISH_DIR (upstream checkout), CPP_ENGINE, RUST_ENGINE,
#              PARITY_TIMEOUT (per-engine-invocation deadline in seconds, default 600).
#
# Usage:
#   scripts/parity.sh                 # perft + state + eval + search (depth 13)
#   scripts/parity.sh perft [depth]
#   scripts/parity.sh eval
#   scripts/parity.sh search [depth]  # bench sequence, per-position info lines
#   scripts/parity.sh standalone [depth] [hash]   # each position independently
#   scripts/parity.sh scan [maxdepth] # find the deepest depth that still matches
#
# Exit code: 0 only if every comparison ran and matched.
#   1  a mismatch was found
#   2  harness misuse (missing/non-executable engine, unknown mode)
#   3  an engine exited non-zero, timed out, or produced no/incomplete records
#      (a failed execution is never treated as agreement)
# Engine stderr is kept in a log and shown on failure.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
CRATE="$(cd "$HERE/.." && pwd)"
# Checkout of upstream Stockfish whose src/ holds the C++ reference: set
# STOCKFISH_DIR, or place stockcrab inside a Stockfish checkout (the default).
ROOT="${STOCKFISH_DIR:-$(cd "$CRATE/.." && pwd)}"
CPP="${CPP_ENGINE:-$ROOT/src/stockfish}"
RUST="${RUST_ENGINE:-$CRATE/target/release/stockcrab}"
export RUST_MIN_STACK=${RUST_MIN_STACK:-50000000}
TIMEOUT_S="${PARITY_TIMEOUT:-600}"

FAILURES=0

die() { echo "ERROR: $*" >&2; exit 2; }
[ -f "$CPP" ] && [ -x "$CPP" ]   || die "C++ engine not found/executable at $CPP (build it: cd $ROOT/src && make -j8 build ARCH=<your-arch>)"
[ -f "$RUST" ] && [ -x "$RUST" ] || die "Rust engine not found/executable at $RUST (build it: cargo build --release --bin stockcrab)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/parity.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SEQ=0
PAIR=0

# engine_failed <name> <what> <log>: a failed execution is a hard error, never a match.
engine_failed() {
  echo "ENGINE FAILURE: $1: $2" >&2
  if [ -s "$3" ]; then echo "--- last lines of $1 stderr/log ---" >&2; tail -n 15 "$3" >&2; fi
  echo "PARITY FAILED (engine failure)"
  exit 3
}

# run_uci <cpp|rust> <commands> <outfile>: feed commands on stdin (which must end in
# `quit`). stdout -> outfile, stderr -> outfile.err. Non-zero exit/timeout is fatal.
run_uci() {
  local name=$1 cmds=$2 out=$3 bin rc=0
  if [ "$name" = cpp ]; then bin=$CPP; else bin=$RUST; fi
  tmo "$TIMEOUT_S" "$bin" <<<"$cmds" >"$out" 2>"$out.err" || rc=$?
  if [ "$rc" -eq 124 ]; then engine_failed "$name" "timed out after ${TIMEOUT_S}s" "$out.err"; fi
  if [ "$rc" -ne 0 ]; then engine_failed "$name" "exited with status $rc" "$out.err"; fi
}

# run_bench <cpp|rust> <outfile> args...: `engine bench ...`, stdout+stderr -> outfile.
run_bench() {
  local name=$1 out=$2 bin rc=0; shift 2
  if [ "$name" = cpp ]; then bin=$CPP; else bin=$RUST; fi
  tmo "$TIMEOUT_S" "$bin" bench "$@" >"$out" 2>&1 || rc=$?
  if [ "$rc" -eq 124 ]; then engine_failed "$name" "bench timed out after ${TIMEOUT_S}s" "$out"; fi
  if [ "$rc" -ne 0 ]; then engine_failed "$name" "bench exited with status $rc" "$out"; fi
}

# Run one UCI script on both engines; outputs in $WORK/cpp.N / $WORK/rust.N.
uci_pair() {
  SEQ=$((SEQ+1)); PAIR=$SEQ
  run_uci cpp  "$1" "$WORK/cpp.$SEQ"
  run_uci rust "$1" "$WORK/rust.$SEQ"
}
cpp_out()  { echo "$WORK/cpp.$PAIR"; }
rust_out() { echo "$WORK/rust.$PAIR"; }

# Strip timing-dependent fields so only deterministic output is compared.
strip_nondet() { sed -E 's/ (nps|time|hashfull) [0-9]+//g'; }

# Bench FENs, read from the C++ source so the two can never drift apart.
bench_fens() {
  python3 - "$ROOT/src/benchmark.cpp" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'const std::vector<std::string> Defaults\s*=\s*\{(.*?)\n\};', src, re.S)
for f in re.findall(r'"([^"]+)"', m.group(1)):
    if '/' in f and (' w ' in f or ' b ' in f):
        print(f)
PY
}

# A small fixed set for perft/state/eval covering castling, en-passant and endgame
# positions, plus one in-check position (where eval prints "none (in check)").
FENS=(
  "startpos"
  "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
  "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
  "fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
  "fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
  "fen 2r3k1/pp2ppbp/3p1np1/q7/2P1P3/1PN1BP2/P2Q2PP/3R1RK1 b - - 0 1"
  "fen 4rrk1/pp1n1pp1/2pb1q1p/3p4/3P4/1P1B1N1P/P1PQ1PP1/2KR3R w - - 0 1"
  # Castling rights and the en-passant square must survive a move replay.
  "startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"
  "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 moves e5d7 e8d7 f3d3 c7c5"
  # Side to move is in check: both engines must print "none (in check)".
  "fen 4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
)
NFENS=${#FENS[@]}

# first_num <file> <egrep-pattern>: trailing number of the first matching line
# ("" when there is none).
first_num() { { grep -iE "$2" "$1" || true; } | { grep -oE '[0-9]+$' || true; } | head -n1; }

do_perft() {
  local depth=${1:-5} fails=0 n=0 f c r
  echo "=== PERFT (movegen / make-unmake / state), depth $depth ==="
  for f in "${FENS[@]}"; do
    n=$((n+1))
    uci_pair "position $f
go perft $depth
quit"
    c=$(first_num "$(cpp_out)"  '^Nodes searched')
    r=$(first_num "$(rust_out)" '^Nodes searched')
    if [ -z "$c" ] || [ -z "$r" ]; then
      printf '  FAIL no perft node count: cpp=%q rust=%q  %s\n' "$c" "$r" "${f:0:46}"; fails=$((fails+1))
    elif [ "$c" = "$r" ]; then
      printf '  OK   %-14s %s\n' "$c" "${f:0:46}"
    else
      printf '  DIFF cpp=%s rust=%s  %s\n' "$c" "$r" "${f:0:46}"; fails=$((fails+1))
    fi
  done
  if [ "$n" -ne "$NFENS" ]; then echo "  FAIL ran $n of $NFENS positions"; fails=$((fails+1)); fi
  echo "perft mismatches: $fails"; FAILURES=$((FAILURES+fails))
}

do_state() {
  # Compares the full internal position state (FEN + zobrist key) after replaying
  # moves. Catches StateInfo lifetime/aliasing bugs and ep/castling errors that
  # perft alone can miss.
  echo "=== POSITION STATE (FEN + zobrist key after move replay) ==="
  local fails=0 n=0 f c r
  for f in "${FENS[@]}"; do
    n=$((n+1))
    uci_pair "position $f
d
quit"
    c=$({ grep -iE '^(Fen|Key):' "$(cpp_out)"  || true; } | tr '\n' ' ')
    r=$({ grep -iE '^(Fen|Key):' "$(rust_out)" || true; } | tr '\n' ' ')
    # Both a Fen: and a Key: record are required from each engine.
    if [[ "$c" != [Ff]en:*[Kk]ey:* || "$r" != [Ff]en:*[Kk]ey:* ]]; then
      printf '  FAIL missing Fen/Key records %s\n    cpp : %s\n    rust: %s\n' "${f:0:56}" "$c" "$r"; fails=$((fails+1))
    elif [ "$c" = "$r" ]; then
      printf '  OK   %s\n' "${f:0:56}"
    else
      printf '  DIFF %s\n    cpp : %s\n    rust: %s\n' "${f:0:56}" "$c" "$r"; fails=$((fails+1))
    fi
  done
  if [ "$n" -ne "$NFENS" ]; then echo "  FAIL ran $n of $NFENS positions"; fails=$((fails+1)); fi
  echo "state mismatches: $fails"; FAILURES=$((FAILURES+fails))
}

do_eval() {
  echo "=== STATIC EVAL (NNUE, full-refresh path) ==="
  local fails=0 n=0 f c r cchk rchk
  for f in "${FENS[@]}"; do
    n=$((n+1))
    uci_pair "position $f
eval
quit"
    c=$({ grep -i 'side to move, internal units' "$(cpp_out)"  || true; } | { grep -oE '[-+][0-9]+' || true; })
    r=$({ grep -i 'side to move, internal units' "$(rust_out)" || true; } | { grep -oE '[-+][0-9]+' || true; })
    cchk=0; rchk=0
    if grep -qi 'none (in check)' "$(cpp_out)";  then cchk=1; fi
    if grep -qi 'none (in check)' "$(rust_out)"; then rchk=1; fi
    if [ "$cchk$rchk" = 11 ] && [ -z "$c$r" ]; then
      # An empty eval is acceptable ONLY when BOTH engines explicitly report that the
      # side to move is in check ("Final evaluation: none (in check)").
      printf '  OK   %-8s %s\n' "(check)" "${f:0:46}"
    elif [ "$cchk$rchk" = 00 ] && [ -n "$c" ] && [ "$c" = "$r" ]; then
      printf '  OK   %-8s %s\n' "$c" "${f:0:46}"
    else
      printf '  DIFF cpp=%s%s rust=%s%s  %s\n' "$c" "$([ "$cchk" = 1 ] && echo '(in check)' || true)" \
        "$r" "$([ "$rchk" = 1 ] && echo '(in check)' || true)" "${f:0:46}"; fails=$((fails+1))
    fi
  done
  if [ "$n" -ne "$NFENS" ]; then echo "  FAIL ran $n of $NFENS positions"; fails=$((fails+1)); fi
  echo "eval mismatches: $fails"; FAILURES=$((FAILURES+fails))
}

# Number of positions in the C++ engine's built-in bench list ("Position: 1/N").
bench_position_count() {
  local f="$WORK/count.out" n
  run_bench cpp "$f" 16 1 1 default depth
  n=$({ grep -oE '^Position: 1/[0-9]+' "$f" || true; } | { grep -oE '[0-9]+$' || true; } | head -n1)
  if [ -z "$n" ] || [ "$n" -le 0 ]; then
    engine_failed cpp "bench printed no 'Position: 1/N' header" "$f"
  fi
  echo "$n"
}

do_search() {
  # The strictest test: every info line (nodes, seldepth, score, PV) of the whole
  # bench sequence. State carries across positions, so this also covers residual
  # TT/history divergence. Info lines and node totals come from ONE invocation per
  # engine, and the complete normalised outputs are compared in both directions.
  local depth=${1:-13} hash=${2:-16}
  echo "=== SEARCH (bench sequence, depth $depth, Hash=$hash, 1 thread) ==="
  local expect co ro cf rf cn rn cp rp d fail=0
  expect=$(bench_position_count)
  co="$WORK/search.cpp"; ro="$WORK/search.rust"
  run_bench cpp  "$co" "$hash" 1 "$depth" default depth
  run_bench rust "$ro" "$hash" 1 "$depth" default depth
  cf="$WORK/search.cpp.info"; rf="$WORK/search.rust.info"
  { grep -E '^info depth' "$co" || true; } | strip_nondet > "$cf"
  { grep -E '^info depth' "$ro" || true; } | strip_nondet > "$rf"
  cn=$(first_num "$co" '^Nodes searched'); rn=$(first_num "$ro" '^Nodes searched')
  cp=$({ grep -cE '^Position: ' "$co" || true; }); rp=$({ grep -cE '^Position: ' "$ro" || true; })
  d=$({ diff "$cf" "$rf" || true; } | { grep -cE '^[<>]' || true; })
  echo "  total nodes : cpp=${cn:-?} rust=${rn:-?}"
  echo "  positions   : cpp=$cp rust=$rp expected=$expect"
  echo "  info lines  : cpp=$(wc -l < "$cf") rust=$(wc -l < "$rf"), differing (both directions)=$d"
  if [ -z "$cn" ] || [ -z "$rn" ]; then echo "  FAIL: missing 'Nodes searched' total"; fail=1; fi
  if [ ! -s "$cf" ] || [ ! -s "$rf" ]; then echo "  FAIL: empty info-line record"; fail=1; fi
  if [ "$cp" != "$expect" ] || [ "$rp" != "$expect" ]; then echo "  FAIL: wrong position count"; fail=1; fi
  if [ "$cn" != "$rn" ] || [ "$d" -ne 0 ]; then fail=1; fi
  if [ "$fail" -eq 0 ]; then
    echo "  EXACT MATCH"
  else
    echo "  MISMATCH -- first differences:"; { diff "$cf" "$rf" || true; } | head -8
    FAILURES=$((FAILURES+1))
  fi
}

do_standalone() {
  # Each position searched independently (fresh TT). Separates genuine per-position
  # search bugs from cross-position residual-state effects.
  local depth=${1:-13} hash=${2:-16} n=0 bad=0 expect f c r
  echo "=== SEARCH STANDALONE (each position independently, depth $depth, Hash=$hash) ==="
  expect=$(bench_position_count)
  bench_fens > "$WORK/fens"
  while IFS= read -r f; do
    if [ -z "$f" ]; then continue; fi
    n=$((n+1))
    uci_pair "position fen $f
bench $hash 1 $depth current depth
quit"
    c=$({ grep -E "^info depth $depth " "$(cpp_out)"  || true; } | strip_nondet)
    r=$({ grep -E "^info depth $depth " "$(rust_out)" || true; } | strip_nondet)
    if [ -z "$c" ] && [ -z "$r" ]; then
      # No depth line at all is legitimate ONLY for a position without legal moves
      # (mate/stalemate: "info depth 0 score ..." + "bestmove (none)"). Require that
      # exact record from both engines; silence or failure is not agreement.
      c=$({ grep -hE '^info depth 0 |^bestmove \(none\)' "$(cpp_out)"  "$(cpp_out).err"  || true; } | strip_nondet)
      r=$({ grep -hE '^info depth 0 |^bestmove \(none\)' "$(rust_out)" "$(rust_out).err" || true; } | strip_nondet)
      case "$c" in *"bestmove (none)"*) ;; *) c="" ;; esac
    fi
    if [ -z "$c" ] || [ -z "$r" ] || [ "$c" != "$r" ]; then
      bad=$((bad+1))
      if [ "$bad" -le 3 ]; then
        printf '  MISMATCH #%d %s\n    cpp : %s\n    rust: %s\n' "$n" "${f:0:50}" "$c" "$r"
      fi
    fi
  done < "$WORK/fens"
  if [ "$n" -ne "$expect" ]; then
    echo "  FAIL: ran $n positions, expected $expect"; bad=$((bad+1))
  fi
  echo "standalone mismatches: $bad of $n"; FAILURES=$((FAILURES+bad))
}

do_scan() {
  local max=${1:-15} hash=${2:-16} last=0 d c r co ro
  echo "=== DEPTH SCAN (deepest exactly-matching bench depth, Hash=$hash) ==="
  co="$WORK/scan.cpp"; ro="$WORK/scan.rust"
  for d in $(seq 1 "$max"); do
    run_bench cpp  "$co" "$hash" 1 "$d" default depth
    run_bench rust "$ro" "$hash" 1 "$d" default depth
    c=$(first_num "$co" '^Nodes searched'); r=$(first_num "$ro" '^Nodes searched')
    if [ -z "$c" ] || [ -z "$r" ]; then
      echo "  depth $d: FAIL no 'Nodes searched' total (cpp=${c:-?} rust=${r:-?})"; FAILURES=$((FAILURES+1)); break
    elif [ "$c" = "$r" ]; then echo "  depth $d: MATCH  $c"; last=$d
    else echo "  depth $d: DIFFER cpp=$c rust=$r"; FAILURES=$((FAILURES+1)); break; fi
  done
  echo "deepest exact-match depth: $last"
}

mode="${1:-all}"
case "$mode" in
  perft)      do_perft "${2:-5}" ;;
  state)      do_state ;;
  eval)       do_eval ;;
  search)     do_search "${2:-13}" "${3:-16}" ;;
  standalone) do_standalone "${2:-13}" "${3:-16}" ;;
  scan)       do_scan "${2:-15}" "${3:-16}" ;;
  all)        do_perft 5; echo; do_state; echo; do_eval; echo; do_search 13 ;;
  *)          die "unknown mode '$mode' (perft|state|eval|search|standalone|scan|all)" ;;
esac

echo
if [ "$FAILURES" -eq 0 ]; then echo "PARITY OK"; exit 0
else echo "PARITY FAILED ($FAILURES mismatch group(s))"; exit 1; fi
