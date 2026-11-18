#!/bin/bash
# Benchmark RIFE in Core ML su Mac. Uso: bash ~/smoovplay/run_coreml.sh  (dopo run_bench.sh)
cd "$(dirname "$0")" || exit 1
exec > >(tee coreml_result.txt) 2>&1
echo "== SmoovPlay: RIFE in Core ML =="
VENV=.venv-mac
[ -x "$VENV/bin/python" ] || { echo "ERRORE: lancia prima bash ~/smoovplay/run_bench.sh"; exit 1; }
UV=$(command -v uv 2>/dev/null)
[ -z "$UV" ] && for u in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do [ -x "$u" ] && UV="$u" && break; done
echo "-> Installo coremltools..."
if [ -n "$UV" ]; then "$UV" pip install --python "$VENV/bin/python" -q coremltools || exit 1
else "$VENV/bin/python" -m pip install -q coremltools || exit 1; fi
echo
"$VENV/bin/python" -W ignore scripts/coreml_mac.py 2>&1 | grep -v -E "Running MIL|Converting PyTorch|scikit-learn|has not been tested"
echo; echo "FATTO. Risultati in ~/smoovplay/coreml_result.txt: puoi tornare su Claude."
