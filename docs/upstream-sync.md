# Upstream sync

The canonical Stockfish revision, network filename, full network checksum, and
bench node count are pinned together in [`config/upstream.env`](../config/upstream.env).
The network comes from [Stockfish's network host](https://tests.stockfishchess.org/api/nn/nn-684d0e60e71d.nnue)
through `scripts/fetch-net.sh`.

## 2026-11-27 — `3e8fde75` to `88337401`

The port includes the search and position changes through `54845c3e` and the
five subsequent commits. The depth-13 bench at Hash 16 and Threads 1 matches
C++ at 1,128,273 nodes and all 639 normalized search records. Perft, position
state, and static evaluation also match. Thirty standard and thirty Chess960
differential positions matched at depth 8.

| Upstream commit | Rust treatment |
|---|---|
| `d911805c` | The C++ compile-time attack-table refactor has no corresponding search change in the Rust table representation. |
| `81c86513` | C++ comment and include cleanup. |
| `d0401bd0` | Cache the bounded repetition lookback window. |
| `65a6fa73` | Remove unused null-stack branches in worker move handling. |
| `29f9cb7b` | C++ Makefile rebuild tracking. Cargo tracks Rust build inputs. |
| `54845c3e` | Allow quiescence futility pruning for promotions. |
| `8d4320f3` | Simplify FEN castling-right parsing and preserve the original invalid token in errors. |
| `7f531dcd` | Split pawn correction history by whether any queen remains. |
| `6ae0e8fb` | C++ custom allocator change. Rust's typed allocations manage array lengths and destruction. |
| `d60859c3` | Pin `nn-684d0e60e71d.nnue` and its full SHA-256. |
| `88337401` | Remove the evaluation correlation term. |
