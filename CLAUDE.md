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
   - chiaki-ng v1.10.1 compilato con il tap (18/11/2026): sorgente in `~/src/chiaki-ng`, app in
     `build/gui/chiaki.app`, 10 simboli `frametap_*`, parte (ripiego su OpenGL: manca MoltenVK, non serve per il tap).
     Bastano i Command Line Tools, niente Xcode. Costo disco ~2 GB brew + 250 MB sorgenti/build. Ricetta:
     `chiaki_tap/build_macos.sh`; avvio: `bash chiaki_tap/run_chiaki_frametap.sh` (serve `CHIAKI_FRAMETAP=1`, che
     Finder/`open` non passano). Trucchi: nanopb vuole il modulo Python `protobuf<6` (venv `~/src/pbvenv`, cmake
     `-DPYTHON_EXECUTABLE`); gli header di openssl@4 coprono quelli di openssl@3, quindi `brew unlink openssl@4`
     durante la compilazione e `brew link openssl@4` dopo (lo fa lo script).
   - **PS5 vera provata (18/11/2026, GTA V)**: registrata, stream H.265 1080p a 60 Hz: 59,8 fotogrammi/s, 30 nuovi/s
     (179 nuovi / 177 ripetuti in 6 s: il rilevamento per dimensione funziona), 0 persi, intervalli di arrivo p50 13,7 /
     p95 25 ms. `play.py --tap` a regime: ~52-55 fps mostrati, AI 22,7 ms, 0-1 in ritardo, 3-4 saltati/s; i primi secondi
     dopo la connessione sono molto peggio (14-25 fps, ~40 in ritardo). chiaki-ng usa ~99% di un core. Se `play.py` parte
     con uno stream già avviato resta a 0 fps: serve riavviare chiaki-ng (o chiudere/riaprire lo stream) con `play.py` in attesa.
     Ctrl+O/Cmd+O non apre il menu in schermo intero: si chiude chiaki-ng e lo si riavvia.
   - **App sul Desktop** (`bash mac/make_app.sh` crea `~/Desktop/SmoovPlay.app`, icona inclusa): apre Terminale con
     `mac/smoovplay_mac.sh`, che pulisce, avvia `play.py --tap`, ASPETTA che abbia caricato Core ML ("ready, waiting for
     the stream" nel log), poi avvia `chiaki stream PS5-590 192.168.1.74` con il tap, e chiude tutto quando uno dei due
     finisce. Nome/IP della PS5 in cima allo script (o `SMOOVPLAY_PS5_NAME`/`_HOST`). Log in `~/Library/Logs/SmoovPlay/`.
     Due lezioni: (1) l'app lanciata direttamente prende `No route to host` verso la PS5 (permesso macOS "Rete locale"
     non effettivo per un'app fatta a mano), mentre da Terminale funziona: per questo passa da `open -a Terminal`
     (`osascript ... do script` no). (2) Il programma AI deve essere pronto PRIMA che parta lo stream: l'intestazione del
     codec arriva una volta sola e, se il modello si carica dopo, l'immagine resta a 0 fps. Per questo in modalità `--tap`
     il modello si carica prima di aspettare lo stream (risoluzione da `--res`, default 1080p).
     Attenzione: `pkill -f` con il nome dello script dentro il comando uccide la propria shell.
   - **Qualità (18/11/2026, dopo il primo giro con l'app)**: trovati e corretti (1) la finestra di chiaki-ng stava SOPRA
     quella AI, con il suo avviso "Renderer Fallback": ora `chiaki --opengl` e la finestra AI è `FLOATING`; (2) il
     lettore scartava migliaia di fotogrammi decodificati (3206 in una sessione) perché su schermi quasi fermi (menu) la
     dimensione dei pacchetti dice "nuovo" per quasi tutti: ora si saltano i fotogrammi con miniatura uguale
     (`STILL_LEVEL`) e si salta l'AI per i fotogrammi già vecchi (`LATE_S`); (3) il decoder non si riprendeva se entrava a
     metà GOP ("Could not find ref"): ora chiede un keyframe se dopo 1 s non esce nessuna immagine; (4) conversione
     diretta a RGB nel lettore (`ChiakiSource.pix_fmt`), niente più flip per fotogramma: corpo del ciclo 24,3 ms.
     Aperto: in gioco ~56-60 fps mostrati, late 0, ma `dropped` ~17-23/s (fotogrammi che cadono nello stesso tick di
     schermo a 240 Hz): serve un ritmo di presentazione regolare. Qualità dell'immagine in sé (bitrate H.265 di chiaki-ng,
     RIFE a movimento dimezzato, artefatti) non ancora valutata a occhio.
   - **Cavo Ethernet (18/11/2026)**: con il Mac in Wi-Fi chiaki-ng misurava 8-11% di pacchetti persi, 438 fotogrammi
     danneggiati e 140 correzioni FEC fallite (= glitch e micro-scatti); col cavo 0 e 0. Il Mac mini ha `en0` Ethernet.
   - **Ritardo adattivo** (`--tap`, `--delay N` lo fissa): il ritardo segue il 98º percentile del tempo di "pronto" dei
     fotogrammi (+6 ms, tra 30 e 150 ms; su veloce, giù 4 ms/s). Il fotogramma reale parte PRIMA di quello AI (`pending`
     ordinato per scadenza). Con `LATE_S` = 26 ms: ritardo ~76 ms (era 80 fisso, 87 con `LATE_S` 40), 67-75 fps mostrati,
     `dropped` ~14/s. Limite vero: GTA qui gira a ~37 fotogrammi nuovi/s e l'AI costa ~25 ms, cioè ~90% di un core
     di calcolo; sotto ~70 ms serve un'AI più veloce (codificare ogni fotogramma una volta sola, punto 2).
   - **HDR**: lo stream di chiaki-ng è H.265 SDR 8 bit, BT.709, range limitato (non HDR); la conversione di PyAV rispetta
     BT.709 (differenza 0,27 su 255 contro 7,05 con BT.601). Se l'immagine appare "strana", la causa non è la matrice colore.
   - **Da fare**: ~~provare con la PS5 vera~~ fatto, ora ritmo/ritardo (punto 2); era: provare con la PS5 vera (Luca): abilitare Remote Play sulla PS5, registrarla in chiaki-ng, scegliere
     1080p, 60 fps, H.265; poi `bash run_play.sh --tap` PRIMA di connettere. Misurare fps mostrati e ritardo. Il filtro "fotogramma nuovo" per giochi a 60 fps veri (menu) non è gestito in `play.py`.
2. Ottimizzazioni: codificare ogni fotogramma reale una volta sola (oggi il modello codifica entrambi a ogni
   chiamata), passare i fotogrammi come CVPixelBuffer/IOSurface invece che PIL, tarare il ritardo (oggi 80 ms).
3. Upscaling con MetalFX (o FSR 1 come shader) e presentazione con Metal.
4. Tenere il lavoro su un branch git (es. `macos-port`), utile per un'eventuale PR upstream.
