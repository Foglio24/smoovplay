#!/bin/bash
# Crea SmoovPlay.app (con icona) sul Desktop: un doppio clic avvia il programma AI + chiaki-ng e li chiude insieme.
#   bash mac/make_app.sh [cartella di destinazione, default ~/Desktop]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/SmoovPlay.app"
TMP="$(mktemp -d)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>SmoovPlay</string>
<key>CFBundleDisplayName</key><string>SmoovPlay</string>
<key>CFBundleIdentifier</key><string>local.smoovplay.mac</string>
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
cat > "$APP/Contents/MacOS/SmoovPlay" <<LAUNCH
#!/bin/bash
exec /bin/bash "$REPO/mac/smoovplay_mac.sh"
LAUNCH
chmod +x "$APP/Contents/MacOS/SmoovPlay"

# Icona: quadrato arrotondato scuro con un triangolo "play" e due frecce che raddoppiano i fotogrammi.
"$REPO/.venv-mac/bin/python" - "$TMP/icon.png" <<'PY'
import sys
from PIL import Image, ImageDraw
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg = Image.new("RGBA", (S, S))
px = bg.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * S)
        px[x, y] = (int(40 + 60 * t), int(30 + 40 * t), int(120 + 100 * t), 255)  # viola-blu
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
SET="$TMP/SmoovPlay.iconset"
mkdir -p "$SET"
for s in 16 32 128 256 512; do
    sips -z $s $s "$TMP/icon.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$APP/Contents/Resources/SmoovPlay.icns"
rm -rf "$TMP"
touch "$APP"
echo "Creata: $APP"
