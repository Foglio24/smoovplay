#!/bin/bash
# Prototipo SmoovPlay per Mac. Uso: bash ~/smoovplay/run_play.sh            (demo 1080p, 30 -> 60 fps)
#                                    bash ~/smoovplay/run_play.sh --res 720p --mult 3   (720p, 30 -> 90 fps)
#                                    bash ~/smoovplay/run_play.sh ~/Movies/video.mp4    (un tuo video)
# Tasti: SPAZIO accende/spegne i fotogrammi AI, F schermo intero, Q esce.
cd "$(dirname "$0")" || exit 1
VENV=.venv-mac
[ -x "$VENV/bin/python" ] || { echo "ERRORE: lancia prima bash ~/smoovplay/run_bench.sh"; exit 1; }
UV=$(command -v uv 2>/dev/null)
[ -z "$UV" ] && for u in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do [ -x "$u" ] && UV="$u" && break; done
if ! "$VENV/bin/python" -c "import glfw, OpenGL, av, PIL, coremltools" 2>/dev/null; then
  echo "-> Installo glfw, PyOpenGL, PyAV, Pillow..."
  if [ -n "$UV" ]; then "$UV" pip install --python "$VENV/bin/python" -q glfw PyOpenGL av pillow coremltools || exit 1
  else "$VENV/bin/python" -m pip install -q glfw PyOpenGL av pillow coremltools || exit 1; fi
fi
"$VENV/bin/python" -W ignore -u mac/play.py "$@" 2>&1 \
  | grep --line-buffered -v -E "Running MIL|Converting PyTorch|scikit-learn|has not been tested" | tee play_result.txt
