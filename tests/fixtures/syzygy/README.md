# KRvK tablebase fixtures

The three-man king-and-rook versus king WDL and DTZ tables, small enough to keep in the
repository so that some Syzygy tests always run. Their SHA-256 digests are in `SHA256SUMS`:

```sh
cd tests/fixtures/syzygy
sha256sum -c SHA256SUMS
```

`tests/uci_session.rs` and the Syzygy unit tests use them. The broader `tests/syzygy_probe.rs`
and `tests/syzygy_search.rs` suites need the full 3-5 man set via `SYZYGY_PATH`
(`scripts/fetch-syzygy.sh`); `3-4-5.sha256` lists its checksums.
