m4r - Apple's MetalFX upscaler and frame generation on AMD Radeon (Linux, Proton)

Proof of concept. Not affiliated with Apple or AMD. D3D12 games under Proton (vkd3d-proton), RDNA2, RDNA3 and RDNA4 GPUs.

INSTALL

1. Extract this zip into the folder with the game's main .exe. It adds dxgi.dll (OptiScaler with MetalFX), OptiScaler.ini,
   the m4r folder, install.sh and uninstall.sh. Games that already have OptiScaler: this dxgi.dll replaces it.
2. Run it there in a terminal:   bash install.sh
   It downloads Apple's MetalFX files (macOS 27.2.1, ~1.2 MiB), checks their SHA-256 hashes and builds m4r's plans for your
   GPU and display (a minute or two the first time; the next game reuses them from ~/.cache/m4r).
   Needs python3 with numpy and LLVM 23 or newer with lld. install.sh names the packages if something is missing:
     Arch, CachyOS:  sudo pacman -S llvm lld python-numpy
     Fedora:         sudo dnf install llvm lld python3-numpy
     Debian, Ubuntu: LLVM 23 from https://apt.llvm.org, then sudo apt install lld-23 python3-numpy
   Already have metalfx-27.2.1.zip? Put it next to install.sh (or bash install.sh --metalfx /path/to/it): nothing is downloaded.
3. Steam: the game's Properties > General > Launch Options:
     WINEDLLOVERRIDES="dxgi=n,b" %command%
4. In the game select DLSS (or FSR or XeSS): OptiScaler runs MetalFX in its place. Insert opens OptiScaler's menu, where
   the upscaler shows as MetalFX.

FRAME GENERATION

OptiScaler.ini is set up for MetalFX FG (FGInput=upscaler, FGOutput=metalfx). Turn frame generation on in OptiScaler's
menu, or with [FrameGen] Enabled=true in OptiScaler.ini.

OTHER RESOLUTIONS

install.sh builds plans for the connected displays and the quality modes Quality, Balanced, Performance and Ultra
Performance. Another display, DLAA or a game with its own render scale:
  bash install.sh --display 3840x2160 --modes quality,balanced,performance,ultra,dlaa --input 1600x900
A missing plan shows in OptiScaler.log (the MetalFX error names the plan); run install.sh with that size.

SETTINGS (OptiScaler.ini, [MetalFX])

FlipJitter=true   for games whose jitter has the other sign (blurry, shimmering detail; Ghost of Tsushima needs it)
FGPacing          frame generation pacing

UNINSTALL

  bash uninstall.sh            (bash uninstall.sh --cache also removes ~/.cache/m4r)

Plans contain Apple's translated kernels and weights: keep them to yourself.
m4r is GPL-3.0-only (LICENSE). Apple's MetalFX files are not covered by that license.
