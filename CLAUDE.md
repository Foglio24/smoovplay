# Progetto: SmoovPlay per Mac (porting sperimentale)

Rispondi in italiano, con tono diretto. Luca procede passo per passo: quando qualcosa si blocca, risolvi il punto preciso.

## Contesto

Questo è un clone di https://github.com/smoovplay/smoovplay: streaming della PS5 (Remote Play tramite chiaki-ng)
con frame generation AI (RIFE 4.22 lite) che porta i giochi da 30 a 60/90/120 fps. Upstream supporta solo
Windows con GPU NVIDIA (CUDA, TensorRT, NVDEC, DXGI). Obiettivo: verificare e costruire una versione per Mac
Apple Silicon. Il lavoro è iniziato in una chat su claude.ai il 18/11/2026; questo file ne riassume lo stato.

Macchina: Mac mini, Apple M5 Pro, macOS 27.2.1, monitor 2560x1440 a 240 Hz. **Poco spazio libero su disco**:
tenerne conto prima di installare cose pesanti (Qt, Homebrew, build di chiaki-ng).

## Ambiente Python

- Usa **`.venv-mac`** (Python 3.12, creato con `uv`): contiene torch 2.15, numpy, opencv-python, coremltools 9.0,
  glfw, PyOpenGL, av, pillow.
- **Non usare `.venv`**: è Python 3.9 di sistema, troppo vecchio.
- Pesi RIFE in `models/flownet_v4.22.lite.pkl`, modelli Core ML generati in `models/coreml/`.

## Risultati misurati (M5 Pro)

Budget: stream a 30 fps → 33,3 ms per fotogramma reale, con il 20% di margine ≤ 26,7 ms.

- **PyTorch MPS** (`scripts/bench_mac.py`, codice upstream invariato con `device="mps", cuda_graph=False`):
  a 1080p motion half costa 29,9 ms per i 60 fps, troppo; a 720p 60 fps sì. Nessun fallback su CPU.
- **Core ML** (`scripts/coreml_mac.py`): 1080p half **22,4 ms** → 60 fps sì; 720p half 11,0 ms → 90 fps;
  120 fps no. **Neural Engine ~20× più lento della GPU**: usare `CPU_AND_GPU`. PSNR 50-55 dB contro il riferimento.
- **Prototipo** (`mac/play.py`): demo 1080p 30→60 fps stabile per 2,5 minuti, 22,2 ms per fotogramma AI,
  0 in ritardo, 8 saltati su ~8.000.

Nota tecnica: per convertire in Core ML i resize con `scale_factor` vanno sostituiti con dimensioni fisse
(il tracing produce aritmetica sulle shape che coremltools non accetta).

## File aggiunti (non in upstream)

- `scripts/bench_mac.py`: benchmark PyTorch MPS
- `scripts/coreml_mac.py`: conversione Core ML, benchmark e controllo dell'immagine (`RifePair` = forward di FastIFNet come nn.Module)
- `mac/rife_coreml.py`: modello Core ML con input/output immagine RGB (padding e crop interni)
- `mac/play.py`: prototipo con due processi (worker: sorgente + Core ML; principale: finestra OpenGL con ritmo
  regolare), memoria condivisa, sorgente demo o file video simulati come stream live. Tasti: SPAZIO AI on/off, F, Q
- `run_bench.sh`, `run_coreml.sh`, `run_play.sh`: launcher (log in `bench_result.txt`, `coreml_result.txt`, `play_result.txt`)
- `mac/ISSUE_DRAFT.md`: proposta (in inglese) da pubblicare come issue upstream

Comandi: `bash run_play.sh` (demo 1080p→60), `bash run_play.sh --res 720p --mult 3` (→90),
`bash run_play.sh ~/Movies/video.mp4`.

## Prossimi passi

1. **PS5 vera**. Fatto (branch `macos-port`, 18/11/2026, non ancora committato):
   - `chiaki_tap/frametap.c`: ramo POSIX (`shm_open` + `mmap`, atomics ARM64, orologio `CLOCK_UPTIME_RAW` a 1e9 Hz,
     nessun evento: il lettore fa polling). Il ramo `_WIN32` è intatto. Nomi: `Local\chiaki-frametap` → `/chiaki-frametap`;
     oltre 31 caratteri (limite macOS) diventa `/smp-<FNV-1a 64>`, uguale in C e Python.
   - `framegen/chiaki_tap.py`: `FrameTap`/`AudioTap`/`qpc()` funzionano anche su POSIX (`_posixshmem`, non
     `multiprocessing.shared_memory`: il resource tracker toglierebbe il segmento).
   - `chiaki_tap/selftest.py` gira su Mac (`.venv-mac/bin/python chiaki_tap/selftest.py`): 40 frame + audio identici byte per byte.
   - `mac/play.py --tap [NOME]`: sorgente live da `ChiakiSource` (decodifica software: HEVC 1080p ~1,4 ms, VideoToolbox non serve).
     Provata con la PS5 finta (`scripts/frametap_fake.py ... --chiaki-like --sw`): AI 22,7 ms, ~56,7 fps mostrati (non 60),
     ~1,5 fotogrammi/s saltati. Misurato: 0 scartati dal lettore, ma la sorgente rende ~29 fotogrammi nuovi/s (non 30) con
     picchi a 47/s: la PS5 finta codifica x265 sulla stessa CPU e il suo ritmo non è regolare. Quindi il calo viene
     dalla sorgente finta, non da Core ML; da riconfermare con la PS5 vera.
   - Cambio di comportamento: su POSIX il segmento shm sopravvive a un crash (su Windows sparisce con l'ultimo handle).
     Dopo un crash di chiaki-ng `magic` resta acceso e `play.py --tap` aspetta uno stream morto: si pulisce con un riavvio
     o `shm_unlink`. `framegen/tap.py` (fotogrammi decodificati) non è stato portato: non serve per la PS5.
   - **Da fare**: compilare chiaki-ng v1.10.1 su macOS con `chiaki_tap/apply_patch.py` (ancora non installato nulla:
     serve prima stimare lo spazio, ~16 GB liberi; capire se bastano i Command Line Tools o serve Xcode completo),
     poi provare con la PS5 vera. Il filtro "fotogramma nuovo" per giochi a 60 fps veri (menu) non è gestito in `play.py`.
2. Ottimizzazioni: codificare ogni fotogramma reale una volta sola (oggi il modello codifica entrambi a ogni
   chiamata), passare i fotogrammi come CVPixelBuffer/IOSurface invece che PIL, tarare il ritardo (oggi 80 ms).
3. Upscaling con MetalFX (o FSR 1 come shader) e presentazione con Metal.
4. Tenere il lavoro su un branch git (es. `macos-port`), utile per un'eventuale PR upstream.
