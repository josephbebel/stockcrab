#!/usr/bin/env bash
# Download the default NNUE network and verify it.
#
# The default network is verified against the full SHA-256 in config/upstream.env.
# An explicitly requested other network uses its filename's 12-digit hash prefix.
#
# Usage: scripts/fetch-net.sh [nn-XXXXXXXXXXXX.nnue]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

. config/upstream.env
name="${1:-$STOCKCRAB_NETWORK}"
[[ "$name" =~ ^nn-[0-9a-f]{12}\.nnue$ ]] || { echo "bad network name: '$name'" >&2; exit 2; }

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d ' ' -f 1; }
verify() {
  local digest
  digest="$(sha256 "$1")"
  if [ "$name" = "$STOCKCRAB_NETWORK" ]; then
    [ "$digest" = "$STOCKCRAB_NETWORK_SHA256" ]
  else
    [ "${digest:0:12}" = "${name:3:12}" ]
  fi
}

if [ -f "$name" ] && verify "$name"; then
  echo "$name already present and verified"
  exit 0
fi

echo "downloading $name ..."
tmp="$(mktemp "${name}.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
curl -fsSL --connect-timeout 30 --max-time 1800 --retry 3 --retry-delay 2 "https://tests.stockfishchess.org/api/nn/$name" -o "$tmp"
verify "$tmp" || { echo "checksum mismatch for $name" >&2; exit 1; }
chmod 644 "$tmp"
mv "$tmp" "$name"
trap - EXIT
echo "saved $name"
