# Binary downloads

Open a successful GitHub Actions CI run for the source revision you want.
Download the artifact for your platform and extract its inner ZIP. The adjacent
`.zip.sha256` file contains the archive checksum.

| Artifact | CPU requirements |
|---|---|
| `stockcrab-linux-x86-64-v3` | x86-64 with AVX2 and BMI2, plus the other x86-64-v3 features |
| `stockcrab-linux-x86-64-baseline` | x86-64 baseline, with slower scalar NNUE |
| `stockcrab-windows-x86-64-v3` | x86-64-v3 on Windows |
| `stockcrab-macos-arm64-dotprod` | Apple Silicon |

Each archive includes a `RUN.txt` with the exact network URL on Stockfish's
server, its full SHA-256, the bench command, and the expected node count.
Download the network from that URL and place it next to the executable.
On Linux or macOS, run `chmod +x stockcrab` if your ZIP extractor leaves
the executable bit unset.

Use the CI archives and checksum files as GitHub Release attachments so the
published binaries are the ones CI smoke-tested. Build from source with
`make build-native` for machine-specific benchmarking. An M1 Mac can
independently check the Apple Silicon archive by running the included bench.
