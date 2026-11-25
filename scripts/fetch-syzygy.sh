#!/usr/bin/env bash
# Download the 3-, 4- and 5-piece Syzygy tablebases (WDL + DTZ, about 1 GB, 290 files)
# from Lichess and verify every file against the published SHA-256 list.
#
# Files are downloaded to `<name>.part.<pid>`, checked, and only then renamed, so an
# interrupted or concurrent run can never leave a truncated table that looks valid
# (the engines would reject or, worse, misread it). Re-running skips verified files.
#
# Usage: scripts/fetch-syzygy.sh [directory]      (default: ./syzygy/3-4-5)
# Then:  SYZYGY_PATH=<directory> cargo test --release --test syzygy_probe
set -euo pipefail

BASE=https://tablebase.lichess.ovh/tables/standard

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# `fetch-syzygy.sh --one <dir> <sums-file> <subdir> <name>`: fetch and verify a single file.
if [ "${1:-}" = "--one" ]; then
  dir=$2 sums=$3 sub=$4 name=$5
  want=$(awk -v n="$name" '$2 == n { print $1 }' "$sums")
  [ -n "$want" ] || { echo "no checksum for $name" >&2; exit 1; }
  if [ -f "$dir/$name" ] && [ "$(sha256 "$dir/$name")" = "$want" ]; then exit 0; fi
  curl -fsSL -o "$dir/$name.part.$$" "$BASE/$sub/$name"
  if [ "$(sha256 "$dir/$name.part.$$")" != "$want" ]; then
    mv -f "$dir/$name.part.$$" "$dir/$name.bad"
    echo "checksum mismatch for $name (kept as $name.bad)" >&2
    exit 1
  fi
  mv -f "$dir/$name.part.$$" "$dir/$name"
  echo "ok $name"
  exit 0
fi

# Resolve the script path and a user-given directory against the caller's cwd before
# changing to the repo root; the default directory is relative to the repo root.
self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
dir_arg="${1:-}"
if [ -n "$dir_arg" ]; then
  mkdir -p -- "$dir_arg"
  dir_arg=$(cd "$dir_arg" && pwd)
fi
cd "$(dirname "$self")/.."
dir="${dir_arg:-syzygy/3-4-5}"
mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
sums="$dir/.sha256"

curl -fsSL -o "$sums" "$BASE/sha256"

list() { # list <subdir> <extension>
  curl -fsSL "$BASE/$1/" | grep -o "href=\"[^\"]*\\.$2\"" | sed 's/^href="//; s/"$//' | sed "s|^|$1 |"
}
{ list 3-4-5-wdl rtbw; list 3-4-5-dtz rtbz; } > "$dir/.files"
echo "$(wc -l < "$dir/.files") files wanted in $dir"

status=0
xargs -P "${JOBS:-4}" -L 1 bash -c '"$0" --one "$1" "$2" "$3" "$4"' "$self" "$dir" "$sums" < "$dir/.files" |
  tail -n 3 || status=$?
rm -f "$dir/.files"

bad=$(find "$dir" -name '*.bad' -o -name '*.part.*' | wc -l)
[ "$bad" -eq 0 ] || { echo "$bad bad or partial files left in $dir" >&2; exit 1; }
[ "$status" -eq 0 ] || { echo "some downloads failed" >&2; exit 1; }
echo "done: $(find "$dir" -name '*.rtb[wz]' | wc -l) tables verified in $dir"
