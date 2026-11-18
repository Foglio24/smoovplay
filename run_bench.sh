#!/bin/bash
# Prepara Python e lancia il benchmark SmoovPlay su Mac. Uso: bash ~/smoovplay/run_bench.sh
cd "$(dirname "$0")" || exit 1
exec > >(tee bench_result.txt) 2>&1
echo "== SmoovPlay: benchmark per Mac =="

# Python >= 3.10 (quello di sistema, 3.9, e' troppo vecchio per PyTorch recente)
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/Current/bin/python3 python3; do
  p=$(command -v "$c" 2>/dev/null) || continue
  if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$p"; break; fi
done
UV=$(command -v uv 2>/dev/null)
[ -z "$UV" ] && for u in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do [ -x "$u" ] && UV="$u" && break; done
if [ -z "$UV" ] && [ -z "$PY" ]; then
  echo "-> Installo uv (scarica Python 3.12, senza permessi da amministratore)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "ERRORE: installazione di uv non riuscita"; exit 1; }
  UV="$HOME/.local/bin/uv"
fi

VENV=.venv-mac
if [ ! -x "$VENV/bin/python" ]; then
  echo "-> Creo l'ambiente Python in $VENV..."
  if [ -n "$UV" ]; then "$UV" venv --python 3.12 "$VENV" || exit 1
  else "$PY" -m venv "$VENV" || exit 1; fi
fi
"$VENV/bin/python" --version

echo "-> Installo PyTorch, NumPy e OpenCV (la prima volta qualche minuto)..."
if [ -n "$UV" ]; then "$UV" pip install --python "$VENV/bin/python" -q torch numpy opencv-python || exit 1
else "$VENV/bin/python" -m pip install -q --upgrade pip && "$VENV/bin/python" -m pip install -q torch numpy opencv-python || exit 1; fi

echo; echo "== Test veloce =="
"$VENV/bin/python" scripts/bench_mac.py --quick || exit 1
echo; echo "== Tabella completa =="
"$VENV/bin/python" scripts/bench_mac.py
echo; echo "FATTO. Risultati salvati in ~/smoovplay/bench_result.txt: puoi tornare su Claude."
