#!/bin/bash
# Registra una sessione lunga di SmoovPlay: ogni secondo fps mostrati, ms per fotogramma AI, ritardo, memoria,
# temperature, frequenze e consumi (powermetrics, chiede la password del Mac).
#   bash ~/smoovplay/run_session.sh --check   prova di 8 secondi, senza gioco
#   bash ~/smoovplay/run_session.sh           aspetta SmoovPlay, registra finché lo chiudi (Q), poi un riassunto
# Risultati in ~/Library/Logs/SmoovPlay/sessions/<data-ora>/
cd "$(dirname "$0")" || exit 1
PY=.venv-mac/bin/python
[ -x "$PY" ] || { echo "ERRORE: manca .venv-mac"; exit 1; }
N=2400  # campioni di powermetrics (1 al secondo): 40 minuti al massimo, poi si ferma da solo
[ "$1" = "--check" ] && N=12
echo "Serve la password del Mac per powermetrics (frequenze, consumi, pressione termica)."
sudo -v || exit 1
sudo powermetrics -i 1000 -b 1 -n "$N" -s cpu_power,gpu_power,thermal -f plist 2>/dev/null \
  | "$PY" -u mac/session_record.py "$@"
