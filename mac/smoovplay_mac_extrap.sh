#!/bin/bash
# SmoovPlay per Mac in modalità "reattiva": il fotogramma AI si calcola dopo l'ultimo fotogramma vero invece che tra
# due (--mode extrap): meno ritardo, immagine un po' meno esatta. Stesso resto di smoovplay_mac.sh.
export SMOOVPLAY_LOG_TAG="_extrap"  # its own logs: running both apps one after the other keeps both runs
export SMOOVPLAY_ARGS="--mode extrap ${SMOOVPLAY_ARGS:-}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoovplay_mac.sh"
