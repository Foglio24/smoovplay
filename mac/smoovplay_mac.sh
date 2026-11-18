#!/bin/bash
# SmoovPlay per Mac: avvia il programma AI (mac/play.py --tap) e chiaki-ng con il frame tap, collegato alla PS5,
# e chiude tutto quando finisce uno dei due. Lanciato dall'app SmoovPlay.app sul Desktop (mac/make_app.sh).
#
# Cambia qui sotto il nome e l'indirizzo della tua PS5 (si leggono nella schermata principale di chiaki-ng).
PS5_NAME="${SMOOVPLAY_PS5_NAME:-PS5-590}"
PS5_HOST="${SMOOVPLAY_PS5_HOST:-192.168.1.74}"
CHIAKI="${SMOOVPLAY_CHIAKI:-$HOME/src/chiaki-ng/build/gui/chiaki.app/Contents/MacOS/chiaki}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv-mac/bin/python"
LOGDIR="$HOME/Library/Logs/SmoovPlay"
mkdir -p "$LOGDIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

notify() { osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"SmoovPlay\" with icon caution" >/dev/null 2>&1; }

[ -x "$PY" ] || { notify "Manca l'ambiente Python (.venv-mac). Lancia prima bash run_bench.sh nella cartella smoovplay."; exit 1; }
[ -x "$CHIAKI" ] || { notify "chiaki-ng non è compilato. Lancia: bash chiaki_tap/build_macos.sh ~/src/chiaki-ng"; exit 1; }

clean() {
    pkill -f "mac/play.py" 2>/dev/null
    pkill -f "chiaki.app/Contents/MacOS/chiaki" 2>/dev/null
    sleep 1
    pkill -9 -f "mac/play.py" 2>/dev/null
    pkill -9 -f "chiaki.app/Contents/MacOS/chiaki" 2>/dev/null
    # i segmenti di memoria condivisa sopravvivono ai crash: tolti, il prossimo avvio non vede uno stream morto
    "$PY" -c "
import _posixshmem
for n in ('/chiaki-frametap', '/chiaki-frametap-audio'):
    try: _posixshmem.shm_unlink(n)
    except FileNotFoundError: pass
" 2>/dev/null
}

clean
trap 'clean' EXIT

cd "$REPO" || exit 1
# Prima il programma AI, che aspetta lo stream; poi chiaki-ng, che si collega alla PS5.
"$PY" -W ignore -u mac/play.py --tap > "$LOGDIR/play.log" 2>&1 &
PLAY=$!
# chiaki-ng parte solo quando il programma AI ha caricato il modello e aspetta lo stream: l'intestazione del codec
# arriva una volta sola, all'inizio dello stream, e se manca l'immagine resta a 0 fps.
for _ in $(seq 1 180); do
    grep -q "ready, waiting for the stream" "$LOGDIR/play.log" 2>/dev/null && break
    kill -0 "$PLAY" 2>/dev/null || { notify "Il programma AI si è fermato. Dettagli: $LOGDIR/play.log"; exit 1; }
    sleep 1
done
CHIAKI_FRAMETAP=1 "$CHIAKI" --exit-app-on-stream-exit stream "$PS5_NAME" "$PS5_HOST" > "$LOGDIR/chiaki.log" 2>&1 &
CHI=$!

# Finché girano tutti e due. Se chiaki-ng finisce (PS5 spenta, connessione persa) o chiudi la finestra AI (Q), si chiude tutto.
while kill -0 "$PLAY" 2>/dev/null && kill -0 "$CHI" 2>/dev/null; do sleep 1; done
if ! kill -0 "$CHI" 2>/dev/null && kill -0 "$PLAY" 2>/dev/null; then
    if ! grep -q "Frame tap on" "$LOGDIR/chiaki.log" 2>/dev/null; then
        notify "Non riesco a collegarmi alla PS5 ($PS5_NAME, $PS5_HOST). È accesa (o in standby) e con Remote Play attivo? Dettagli: $LOGDIR/chiaki.log"
    fi
fi
