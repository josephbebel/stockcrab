# Stockcrab build shortcuts (Linux/macOS). Cargo does the actual work.

CARGO ?= cargo
HOST_ARCH := $(shell uname -m)

# The target CPU is passed through RUSTFLAGS; a RUSTFLAGS set by the caller is respected.
NATIVE_FLAGS = RUSTFLAGS="$${RUSTFLAGS:--C target-cpu=native}"
V3_FLAGS = RUSTFLAGS="$${RUSTFLAGS:--C target-cpu=x86-64-v3}"

.PHONY: help net build build-native build-x86-64-v3 test

help:
	@echo "  make net                 Download and verify the default NNUE network"
	@echo "  make build               Release build for the host's portable baseline"
	@echo "  make build-native        Release build for this exact CPU"
	@echo "  make build-x86-64-v3     Release build for AVX2/BMI2 x86-64 CPUs (x86-64 hosts only)"
	@echo "  make test                Run the test suite (release mode)"

net:
	scripts/fetch-net.sh

build:
	$(CARGO) build --release

build-native:
	$(NATIVE_FLAGS) $(CARGO) build --release

build-x86-64-v3:
ifneq ($(filter x86_64 amd64,$(HOST_ARCH)),)
	$(V3_FLAGS) $(CARGO) build --release
else
	@echo "build-x86-64-v3 needs an x86-64 host (this is $(HOST_ARCH))" >&2; exit 1
endif

test: net
	$(CARGO) test --release
