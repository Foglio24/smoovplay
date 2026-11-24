# SmoovPlay for macOS (experimental)

Plays your PS5 over Remote Play and raises 30 fps to 60 fps with RIFE, running in Core ML on the Apple Silicon GPU.
Tested on an M5 Pro, 1080p, macOS 27. Needs Apple Silicon (M1 or newer) and macOS 13+.

## Install

1. Install [Homebrew](https://brew.sh) and the Xcode command line tools (`xcode-select --install`) if you don't have them.
2. Unpack the zip and double-click **Install SmoovPlay.command**.
   macOS may refuse an unsigned script from the internet: right-click it, choose **Open**, then **Open** again.
   Or open Terminal and run `bash "<unpacked folder>/Install SmoovPlay.command"`.
3. Wait. The first install builds chiaki-ng (10 to 30 minutes) and downloads PyTorch (about 1 GB).
4. When it asks, enter your PS5's name and IP address. Register the PS5 in chiaki-ng first (the installer prints the path).

The installer puts everything in `~/.smoovplay` and two apps in `~/Applications`. It is safe to run again.

## Use

- **SmoovPlay**: interpolation. Best picture, about 50 to 60 ms of added delay.
- **SmoovPlay Reattivo**: extrapolation. About 30 ms less delay, picture a little less exact.

On first start macOS asks whether Terminal may use the local network. Say yes: without it the PS5 is not reachable.
Press **Q** in the SmoovPlay window to quit. Logs are in `~/Library/Logs/SmoovPlay`.

Change the PS5 later in `~/.smoovplay/config`.

## Uninstall

Delete `~/.smoovplay`, `~/Applications/SmoovPlay*.app` and `~/Library/Logs/SmoovPlay`.

## Notes

- The apps are made on your own Mac by the installer, so they are not quarantined and need no notarization.
- chiaki-ng is a separate GPL-3.0 project (https://github.com/streetpea/chiaki-ng). The installer downloads it
  and applies the frame-tap patch from `chiaki_tap/`; it is not bundled in this zip.
