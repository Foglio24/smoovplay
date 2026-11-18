#!/bin/bash
# Crea sul Desktop due app (con icona): un doppio clic avvia il programma AI + chiaki-ng e li chiude insieme.
#   SmoovPlay.app           interpolazione (qualità migliore, ~50-60 ms di ritardo)
#   SmoovPlay Reattivo.app  estrapolazione (--mode extrap: ~30 ms di ritardo in meno, immagine un po' meno esatta)
#   bash mac/make_app.sh [cartella di destinazione, default ~/Desktop]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
TMP="$(mktemp -d)"

# build_app "Nome" "script da lanciare" "id" "colore dell'icona (blue|green)"
build_app() {
local NAME="$1" SCRIPT="$2" BID="$3" TINT="$4"
local APP="$DEST/$NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>$NAME</string>
<key>CFBundleDisplayName</key><string>$NAME</string>
<key>CFBundleIdentifier</key><string>$BID</string>
<key>CFBundleExecutable</key><string>SmoovPlay</string>
<key>CFBundleIconFile</key><string>SmoovPlay</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
<key>NSHighResolutionCapable</key><true/>
<key>NSLocalNetworkUsageDescription</key><string>SmoovPlay si collega alla tua PS5 sulla rete locale.</string>
</dict></plist>
PLIST

# Il lanciatore punta allo script nella cartella del progetto, così un aggiornamento dello script vale subito.
# Parte da Terminale: macOS lascia raggiungere la PS5 (rete locale) a Terminale, ma blocca i programmi lanciati da
# un'app creata a mano ("No route to host"). Con "open -a Terminal" lo script gira nella finestra di Terminale, che si
# chiude da sola quando lo script finisce.
cat > "$APP/Contents/MacOS/SmoovPlay" <<LAUNCH
#!/bin/bash
exec /usr/bin/open -a Terminal "$REPO/mac/$SCRIPT"
LAUNCH
chmod +x "$APP/Contents/MacOS/SmoovPlay"

# Icona: quadrato arrotondato scuro con un triangolo "play" e due frecce che raddoppiano i fotogrammi.
"$REPO/.venv-mac/bin/python" - "$TMP/icon.png" "$TINT" <<'PY'
import sys
from PIL import Image, ImageDraw
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg = Image.new("RGBA", (S, S))
px = bg.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * S)
        px[x, y] = ((int(40 + 60 * t), int(30 + 40 * t), int(120 + 100 * t), 255) if sys.argv[2] == "blue"  # viola-blu
                    else (int(20 + 30 * t), int(110 + 80 * t), int(100 + 60 * t), 255))  # verde acqua: modo reattivo
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((64, 64, S - 64, S - 64), radius=210, fill=255)
img.paste(bg, (0, 0), mask)
d = ImageDraw.Draw(img)
d.polygon([(380, 290), (380, 734), (760, 512)], fill=(255, 255, 255, 255))
for i, a in enumerate((0.55, 0.3)):  # fotogrammi "in più": contorni che seguono il triangolo
    o = 70 * (i + 1)
    d.polygon([(380 - o, 290), (380 - o, 734), (760 - o, 512)], outline=(255, 255, 255, int(255 * a)), width=14)
img.save(sys.argv[1])
PY
SET="$TMP/$TINT.iconset"
mkdir -p "$SET"
for s in 16 32 128 256 512; do
    sips -z $s $s "$TMP/icon.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$APP/Contents/Resources/SmoovPlay.icns"
codesign --force --deep -s - "$APP" 2>/dev/null || true  # firma "ad hoc": macOS lancia l'app senza avvisi strani
touch "$APP"
echo "Creata: $APP"
}

build_app "SmoovPlay" "smoovplay_mac.sh" "local.smoovplay.mac" blue
build_app "SmoovPlay Reattivo" "smoovplay_mac_extrap.sh" "local.smoovplay.mac.extrap" green
rm -rf "$TMP"
