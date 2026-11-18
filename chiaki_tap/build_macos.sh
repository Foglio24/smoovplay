#!/bin/bash
# Build chiaki-ng with the frame tap on macOS (Apple Silicon). Tested with chiaki-ng v1.10.1, macOS 27, Homebrew.
#
#   bash chiaki_tap/build_macos.sh <chiaki-ng source> [protobuf venv]
#
# The source must be a v1.10.1 checkout with submodules:
#   git clone --depth 1 --branch v1.10.1 --recurse-submodules --shallow-submodules \
#       https://github.com/streetpea/chiaki-ng.git
# Mirrors chiaki-ng's own macOS CI (.github/workflows/build-macos-arm.yml), plus apply_patch.py. The result is
# <source>/build/gui/chiaki.app, run from the build tree with run_chiaki_frametap.sh (not deployed with macdeployqt).
set -euo pipefail

SRC="${1:?usage: build_macos.sh <chiaki-ng source> [protobuf venv]}"
VENV="${2:-$HOME/src/pbvenv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$(brew --prefix)"

step() { echo; echo "=== $*"; }

step "brew packages"
brew install qt ffmpeg pkgconf opus openssl@3 cmake ninja nasm protobuf@29 speexdsp libplacebo json-c miniupnpc libevent

step "Python protobuf (nanopb's generator needs it; protobuf 5 to match protoc 29)"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv -q --python 3.12 "$VENV"
    uv pip install -q --python "$VENV/bin/python" 'protobuf>=5,<6'
fi

step "frame tap patch"
python3 "$HERE/apply_patch.py" "$SRC"

step "configure + build"
export PATH="$VENV/bin:$B/bin:$PATH" CPATH="$B/opt/ffmpeg/include"
cd "$SRC"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHIAKI_ENABLE_CLI=OFF -DCHIAKI_ENABLE_STEAMDECK_NATIVE=OFF \
    -DPYTHON_EXECUTABLE="$VENV/bin/python" \
    -DOPENSSL_ROOT_DIR="$B/opt/openssl@3" -DOPENSSL_INCLUDE_DIR="$B/opt/openssl@3/include" \
    -DOPENSSL_CRYPTO_LIBRARY="$B/opt/openssl@3/lib/libcrypto.dylib" -DOPENSSL_SSL_LIBRARY="$B/opt/openssl@3/lib/libssl.dylib" \
    -DCMAKE_PREFIX_PATH="$B/opt/openssl@3;$B/opt/qt;$B/opt/protobuf@29"

# OpenSSL 4's headers in $B/include would shadow openssl@3's (the link then misses OPENSSL_sk_set_*_thunks), so
# openssl@4 is unlinked while compiling, like the CI does, and linked again afterwards.
relink=0
if brew list --versions openssl@4 >/dev/null 2>&1 && [[ -e "$B/include/openssl" ]]; then
    brew unlink openssl@4 && relink=1
fi
trap '[[ $relink == 1 ]] && brew link openssl@4 >/dev/null || true' EXIT
cmake --build build --config Release --target chiaki

echo
echo "done: $SRC/build/gui/chiaki.app (start it with chiaki_tap/run_chiaki_frametap.sh)"
