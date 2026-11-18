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
     **Bug trovato in prova (18 nov)**: il filtro "immagine ferma" confrontava ogni fotogramma solo con l'ultimo
     *mostrato* (soglia `STILL_LEVEL` 0,35): un movimento lento (camera ferma-ish, personaggio in idle) cambia ~0,08 per
     immagine, quindi passava 1 immagine su ~5: 6 fps invece di 30 (misurato con una panoramica sintetica; dal vivo 0-13
     fps mostrati con flusso PS5 stabile a 60/s, `frames_lost 0`, quindi non era la rete). Ora `StillFilter` confronta
     anche con il fotogramma *appena ricevuto* (`STEP_LEVEL` 0,04: un ripetuto differisce ~0) e tiene la somma 0,35.
     Test: `tests/test_still_filter.py`. **Non verificato dal vivo**: 0,04 presuppone che i ripetuti del PS5 (HEVC)
     differiscano meno di così; se `fps shown` resta a 60 su schermo fermo e l'AI va in ritardo, alzare `STEP_LEVEL`.
     **Ciclo di presentazione (18 nov)**: `dropped` 20-30/s non era la jitter d'arrivo ma `swap_interval(1)`: su macOS
     `swap_buffers` bloccava 8-11 ms (misurato con un benchmark GL: ~95 giri/s su schermo a 240 Hz; l'upload della
     texture costa 1-2 ms e non c'entra). Il log offriva ~105 fotogrammi/s (shown+dropped) contro 60-75 mostrati; i
     fotogrammi dovuti nello stesso giro venivano scartati tranne l'ultimo. Ora `swap_interval(0)`, si disegna solo
     quando c'è un fotogramma dovuto (o la finestra va ridisegnata) e il ciclo dorme fino al prossimo (<= 2 ms);
     `--vsync` ripristina il vecchio ciclo. Con la clip demo: 60,0 fps fissi (prima 57,7-60,7). **Non verificato dal
     vivo con la PS5**: atteso `dropped` ~0 e movimento più regolare; se si vede tearing o la finestra non si aggiorna,
     provare `--vsync`.
     **Estrapolazione (`--mode extrap`, 18 nov)**: l'interpolazione ha un pavimento di ritardo (il fotogramma AI sta a
     metà tra due reali ma serve il successivo: ~mezzo intervallo + calcolo = 17 + 15 ms, più la jitter d'arrivo: da
     qui il ritardo adattivo di 60-80 ms). L'estrapolazione disegna il fotogramma *dopo* l'ultimo reale, dal moto
     tra gli ultimi due (`RifePair` variante `extrap`: flusso a t=0.5, poi `warp(img1, -t * (f1 - f0 - bias))`, `t` =
     frazione di un intervallo di gioco), quindi non aspetta nulla. Misura (`scripts/extrapolation_probe.py`, clip
     demo con movimento doppio del live, contro il fotogramma vero): ripetere l'ultimo 23,7 dB, estrapolare 35,3 dB,
     interpolare 38,6 dB (caso peggiore 10%: 19,8 / 31,2 / 36,2). Traslazioni pure: spostamento giusto entro 0,15 px;
     **coppie identiche: il flusso dà un moto fantasma di (-0,034; -0,403) px (sd 0,01), che spostava lo schermo fermo di
     0,2 px = tremolio: sottratto (`ext_bias`), ora 0,06 px**. Il ricampionamento del moto sulla griglia di arrivo non
     serve (+0,05 dB). Core ML fp16 13,9 ms (non più veloce: la rete del flusso domina), 49 dB da PyTorch. Con la clip a
     ritardo fisso 30 ms: interp `late` 5-15/s e `ready` 33-49 ms, extrap `late` 0 e `ready` 3-11 ms; col flusso finto
     il ritardo adattivo va da ~50 a 15 ms (il minimo di `min_delay`). Nella finestra un fotogramma estrapolato che
     cadrebbe dopo un reale più nuovo si scarta (`is_stale`/`drop_stale`, `tests/test_extrap_order.py`, contatore
     `stale`); il tempo del fotogramma è `pts + t * (gap tipico)`, non la jitter dell'arrivo. **Costo: l'immagine
     cambia di colpo quando il moto cambia (frenata, scatto): il reale successivo "corregge". Non verificato dal
     vivo con la PS5**; `--mult 3` (t=1/3, 2/3) non validato. Default resta `interp`; sul Desktop ci sono
     `SmoovPlay.app` (interp) e `SmoovPlay Reattivo.app` (extrap, icona verde) per il confronto (`mac/make_app.sh`).
     Interfaccia fissa sopra movimento (sonda, patch incollate identiche in tutti i fotogrammi): non si sposta (scarto
     medio 0,04-0,11 px, peggiore 0,49) ma nel caso peggiore si ammorbidisce un po' (PSNR peggiore 24-30 dB contro 36-38
     di interp: il flusso "perde" 0,3-0,6 px dentro la patch). Le patch erano texture del mondo, non una vera barra HUD:
     da guardare a occhio su HUD e testo. Una coppia che copre k fotogrammi di gioco usa `t / k` (k arrotondato, max 4).
     I log delle due app sono separati (`play.log`/`play_extrap.log`, `SMOOVPLAY_LOG_TAG`).
     **Il gioco non è a 30 fps (18 nov, 30 Mbps)**: alzato il bitrate di chiaki-ng a 30 Mbps
     (`defaults write com.chiaki.Chiaki settings.bitrate_local_ps5 -int 30000`; default 15 Mbps; per tornare
     `defaults delete ...`): immagine più pulita, ma fps mostrati 90-100, ritardo 56 ms, 97 AI saltati. Le righe `tap:`
     (statistiche al secondo) hanno mostrato ~55 fotogrammi nuovi al secondo con passo mediano 14-20 (scena veloce):
     il gioco girava a ~55-60 fps, non a 30 raddoppiati. Il rilevatore dei ripetuti dalla *dimensione* del frame
     compresso (`SizeRepeatDetector`) sbaglia: con una clip a 60 fps tutta di fotogrammi nuovi ne dava ~30/s per
     ripetuti (i frame "intermedi" sono più piccoli); a 15 Mbps scartava frame veri e faceva credere a un gioco a 30.
     Ora `TapSource` non si fida più della dimensione (solo contata nelle righe `tap:`): decide `StillFilter` dai
     pixel, con soglia sui ripetuti `max(STEP_LEVEL, STEP_REL * velocità della scena)` (un ripetuto a bitrate alto ha
     rumore ~0,06 sulla miniatura, due immagini diverse di una scena veloce 5-20). **L'AI ora si accende solo sotto
     44 fps del gioco e si spegne sopra 50** (`AI_ON_FPS`/`AI_OFF_FPS`, fps contati in 2 s): a 55-60 fps l'AI da 15 ms
     per frame occupa l'83% del tempo e aggiunge ritardo senza dare molto; col flusso finto a 60 fps ritardo adattivo
     25-33 ms invece di ~50. Pavimento del ritardo adattivo 30 -> 20 ms. Test: `tests/test_still_filter.py`. **Non
     verificato con la PS5 dopo la modifica**: da guardare se a ~55 fps reali senza AI l'immagine sembra meno fluida
     di prima (per riaccendere l'AI sempre: `AI_OFF_FPS` alto).
     **Provato dal vivo con la PS5 (18 nov): extrap scartato.** Input più rapido (ritardo adattivo ~33 ms contro ~49 di
     interp) ma "sembra di giocare a 30 fps". Nel log gli fps mostrati erano 66-83 e `late` 0, quindi non era la
     finestra; `stale` ~7/s (fino a 14): circa un fotogramma AI su cinque cadeva dopo il reale successivo e veniva
     scartato, e 12 AI saltati per ritardo. Causa non verificata (ipotesi: moto sottostimato su spostamenti grandi, o
     cadenza irregolare per gli scarti). Il default resta `interp`. Stesso giro con interp: 70 fps, ritardo 49 ms,
     `late` 0, `dropped` 0 (prima 20-30/s: ciclo senza vsync + StillFilter verificati). Per riprovare extrap servirebbe
     capire prima da dove vengono gli `stale` (gap tipico contro jitter d'arrivo).
     Se `extrap` piace ma si vede un tremolio su schermo fermo, rimisurare `MOTION_BIAS` con la sonda e cambiare
     `ext_bias` in `scripts/coreml_mac.py` (e il nome del pacchetto).
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
2. Ottimizzazioni. FATTO (18/11/2026): modello Core ML più leggero. Misurato (Apple M5 Pro, 1080p half, fotogrammi reali):
   `base` 21,8 ms; `fold` (beta fuso nei pesi) 21,8 ms, identico: Core ML lo faceva già, la stima dei costi di
   `MLComputePlan` non è un profilo reale; `lean` 17,5 ms (nessun giro a piena risoluzione di maschera/feature, solo il
   flusso sale a piena risoluzione) con 65-67 dB da `base`; `lean2` 14,2 ms (anche i warp intermedi alla risoluzione del
   blocco successivo, solo l'ultimo a piena) con 45,6-47,7 dB da `base`. Predefinito `--model lean3` (vedi sotto); modelli in
   `models/coreml/*_lean2.mlpackage` (nome con suffisso: un pacchetto esistente con lo stesso nome viene riusato, quindi un
   grafo cambiato ha bisogno di un nome nuovo). In-app con la PS5 finta: fotogramma AI 15,5 ms (era 24), ritardo
   adattivo ~60 ms (era 76-87). Codificare ogni fotogramma una volta sola NON serve: l'encoder è ~5% del lavoro
   (~0,5 ms), l'avevo stimato male. Il costo vero è lo spostamento dati (slice, resize, warp), non le convoluzioni
   (~7-15%). Non provati: 720p, `--mult 3`, e giochi reali a ~37 fps con lean2.
   **Tremolio su immagini ferme (18/11/2026)**: il fotogramma AI risultava spostato di ~0,17 px in alto e ~0,05-0,1 px a
   sinistra rispetto ai reali (misurato con traslazioni pure e con coppie identiche; costante in tutta l'immagine e con
   qualsiasi movimento; già presente in PyTorch fp32, -0,12 px, quindi è del modello, non di Core ML; tenere le
   coordinate di warp a 32 bit, variante `lean2g`, non cambia nulla e costa 1 ms). Alternato 30 volte al secondo con i
   fotogrammi reali si vede come un tremolio. `lean3` = `lean2` + offset di campionamento (`RifePair.bias`) nell'ultimo
   warp: errore residuo ~0,03 px, 14,3 ms, 44-46 dB da `base` (che ha lo stesso scarto). Inoltre `GAP_S` (90 ms): niente
   fotogramma AI tra due fotogrammi lontani nel tempo (uno schermo fermo che riparte produceva un fotogramma "fantasma"
   mescolato con uno vecchio di secondi). NON ancora provato dal vivo: prima dell'uso controllare a occhio su testo/HUD
   fermi; l'offset si cambia in `RifePair.bias` (scripts/coreml_mac.py) e il pacchetto va rifatto con un nome nuovo.
   Ancora da fare: CVPixelBuffer/IOSurface (le conversioni PIL costano <1 ms: poco da guadagnare).
3. Upscaling con MetalFX (o FSR 1 come shader) e presentazione con Metal.
4. Tenere il lavoro su un branch git (es. `macos-port`), utile per un'eventuale PR upstream.
