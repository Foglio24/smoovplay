#!/bin/bash
# Start chiaki-ng (built by build_macos.sh) with the frame tap on, so SmoovPlay's `mac/play.py --tap` receives every
# video frame. Without CHIAKI_FRAMETAP=1 this build behaves exactly like stock chiaki-ng.
# The variable has to be set here: an app started from Finder or `open` does not get it.
#   bash chiaki_tap/run_chiaki_frametap.sh [~/src/chiaki-ng]
SRC="${1:-$HOME/src/chiaki-ng}"
export CHIAKI_FRAMETAP=1
exec "$SRC/build/gui/chiaki.app/Contents/MacOS/chiaki"
