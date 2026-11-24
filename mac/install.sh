#!/bin/bash
# SmoovPlay for macOS: one-time installer. Run it from the unpacked release (or from a clone of the repo):
#   bash mac/install.sh
# It copies SmoovPlay to ~/.smoovplay, makes the Python environment, builds chiaki-ng with the frame tap, converts the
# Core ML model, asks for your PS5, and puts SmoovPlay.app and "SmoovPlay Reattivo.app" in ~/Applications.
# Safe to run again: finished steps are skipped. Needs Apple Silicon, macOS 13+, Homebrew and the Xcode command line tools.
#
# Non-interactive use: SMOOVPLAY_PS5_NAME=PS5-123 SMOOVPLAY_PS5_HOST=192.168.1.20 bash mac/install.sh
# Other variables: SMOOVPLAY_HOME (install folder, default ~/.smoovplay), SMOOVPLAY_APPS (where the apps go, default ~/Applications).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${SMOOVPLAY_HOME:-$HOME/.smoovplay}"
APPS="${SMOOVPLAY_APPS:-$HOME/Applications}"
CHIAKI_TAG="v1.10.1"   # the version chiaki_tap/apply_patch.py is written for
step() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Checking this Mac"
[ "$(uname -s)" = Darwin ] || die "this is the macOS installer."
[ "$(uname -m)" = arm64 ] || die "SmoovPlay for Mac needs Apple Silicon (M1 or newer)."
[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 13 ] || die "macOS 13 or newer is needed."
xcode-select -p >/dev/null 2>&1 || die "install the Xcode command line tools first: xcode-select --install"
command -v brew >/dev/null 2>&1 || [ -x /opt/homebrew/bin/brew ] || die "install Homebrew first: https://brew.sh"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
command -v git >/dev/null 2>&1 || die "git is missing (xcode-select --install)."
echo "ok"

step "Copying SmoovPlay to $HOME_DIR/smoovplay"
DEST="$HOME_DIR/smoovplay"
mkdir -p "$HOME_DIR"
if [ "$SRC" != "$DEST" ]; then
    mkdir -p "$DEST"
    # --delete would remove the venv and models; those two folders are kept out of the sync on both sides.
    rsync -a --delete --exclude '/.venv-mac' --exclude '/models' --exclude '.git' --exclude '.DS_Store' "$SRC/" "$DEST/"
fi
cd "$DEST"

step "Python environment"
command -v uv >/dev/null 2>&1 || { echo "installing uv with Homebrew"; brew install uv; }
[ -x .venv-mac/bin/python ] || uv venv -q --python 3.12 .venv-mac
uv pip install -q --python .venv-mac/bin/python -r mac/requirements-mac.txt
.venv-mac/bin/python -c "import torch, coremltools, glfw, OpenGL, av, cv2, PIL; print('packages ok, torch', torch.__version__)"

step "chiaki-ng with the frame tap (the first build takes 10-30 minutes)"
CHIAKI_SRC="$HOME_DIR/chiaki-ng"
CHIAKI_BIN="$CHIAKI_SRC/build/gui/chiaki.app/Contents/MacOS/chiaki"
if [ -x "$CHIAKI_BIN" ]; then
    echo "already built: $CHIAKI_BIN"
else
    [ -d "$CHIAKI_SRC/.git" ] || git clone --depth 1 --branch "$CHIAKI_TAG" --recurse-submodules --shallow-submodules \
        https://github.com/streetpea/chiaki-ng.git "$CHIAKI_SRC"
    bash chiaki_tap/build_macos.sh "$CHIAKI_SRC" "$HOME_DIR/pbvenv"
fi
[ -x "$CHIAKI_BIN" ] || die "chiaki-ng did not build: see the messages above."

step "Core ML model (downloads the weights, converts for 1080p; about a minute)"
.venv-mac/bin/python -W ignore - <<'PY'
import sys
sys.path.insert(0, "mac")
from rife_coreml import convert
from framegen.interpolator import ensure_weights
ensure_weights("4.22.lite")
for variant in ("lean3", "extrap"):
    print("ready:", convert(1920, 1080, 0.5, variant).name)
PY

step "Your PS5"
NAME="${SMOOVPLAY_PS5_NAME:-}" HOST="${SMOOVPLAY_PS5_HOST:-}"
if [ -z "$NAME" ] || [ -z "$HOST" ]; then
    echo "Open chiaki-ng once and register the PS5 there (it shows the console's name and address):"
    echo "  $CHIAKI_BIN"
    echo
    [ -n "$NAME" ] || read -r -p "PS5 name as shown in chiaki-ng (e.g. PS5-123): " NAME
    [ -n "$HOST" ] || read -r -p "PS5 IP address (e.g. 192.168.1.20): " HOST
fi
[[ "$HOST" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "'$HOST' is not an IPv4 address."
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "'$NAME' has characters other than letters, digits . _ -"
cat > "$HOME_DIR/config" <<CFG
PS5_NAME="$NAME"
PS5_HOST="$HOST"
CHIAKI="$CHIAKI_BIN"
CFG
echo "saved $HOME_DIR/config"

step "Apps in $APPS"
mkdir -p "$APPS"
bash mac/make_app.sh "$APPS"

cat <<DONE

Done. Open SmoovPlay from $APPS (or Launchpad).
First time: macOS asks to let Terminal use the local network (say yes), and chiaki-ng must have registered your PS5.
Logs: ~/Library/Logs/SmoovPlay
DONE
