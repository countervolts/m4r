# m4r (metalfx 4 radeon)

want dlss on radeon? check out [d4r](https://github.com/countervolts/d4r)! xess xmx? [x4r](https://github.com/countervolts/x4r)!

m4r runs Apple's **MetalFX** upscaler (the ML temporal scaler, "BBR") and MetalFX frame
interpolation in Windows games on AMD Radeon GPUs under Linux and Proton. The game asks for DLSS/FSR/XeSS as usual;
patches for [OptiScaler](https://github.com/optiscaler/OptiScaler) add **MetalFX** to its upscaler menu and **MetalFX FG**
to its frame generation menu, and hand the frame to m4r ([docs/optiscaler.md](docs/optiscaler.md)). m4r ships the patches,
not an OptiScaler build.

m4r runs Apple's own kernels and network weights, as d4r does with NVIDIA's and x4r with Intel's: the Metal
compute kernels in your copy of MetalFX's `default.metallib` are translated from their AIR (LLVM) bitcode to
AMDGPU code objects and run inside the game's Vulkan command stream through d4r's **vkhip** loader. No network
is re-trained, approximated or replaced.

> **Work in progress / proof of concept.** Not affiliated with Apple or AMD.

## Status

| Piece | State |
|---|---|
| `default.metallib` reader (functions, AIR bitcode) | works, all 90 functions of macOS 27.2.1 |
| AIR → AMDGPU translator (`air/`) | 80 of the 90 functions of MetalFX 27.2.1 translate for gfx1101 (default settings); the rest are the spatial scaler's fragment shaders, a vertex shader and V3/V4's two tile kernels (`unified_pass_tile`, `unified_low_resolution_tile`: struct metadata the parser does not handle yet). The frame interpolation's tile kernel (`warpImage`) works |
| Apple MXU matrix instructions (`llvm.agx3.igemm/fgemm`) | lane layout measured; the integer GEMMs run on RDNA3 WMMA and match a reference convolution exactly, the fp16 ones keep an exact `fdot2` chain ([docs/mxu.md](docs/mxu.md)) |
| Runtime (Vulkan + vkhip, Metal binding model, pass plans) | works: translated `copyTexture` is bit-exact on an RX 7700 XT |
| ML temporal scaler (BBR: tBBRNet + sBBRNet) | **works offline**: full 27-pass pipeline, 640x360 → 1280x720 at 28.27 dB vs a supersampled reference (bilinear 21.9 dB), bit-identical from run to run, 1.7 ms per frame |
| OptiScaler "MetalFX" upscaler (`patches/optiscaler/`) | **works in a test harness under Proton** (vkd3d-proton): 28.27 dB, about 2.7 ms per frame with submit and wait; not tried in a game yet |
| OptiScaler "MetalFX FG" output | **works in a test harness under Proton**: real frames restored bit-exactly, generated frames about 24.5–25.1 dB against the true in-between frame (blend 21.7 dB); not tried in a game yet |
| Frame interpolation | **works offline**: 30 passes, 31.9–32.1 dB vs true in-between frames (frame blend 22.3 dB), 0.87 ms at 1280x720 |
| Temporal V4 (the other ML upscaler with a Neural Engine variant), V3, spatial scaler, denoiser | not started; V4 is a different network (12x12 stride-4 stem, three stride-2 convolutions, four fused 1x1 layers), see [docs/metalfx-bbr.md](docs/metalfx-bbr.md) for how MetalFX picks between them |

## What you need

- Linux, an RDNA3 GPU (developed on an RX 7700 XT, gfx1101), Mesa RADV with `VK_EXT_descriptor_buffer`.
- LLVM with the AMDGPU target and lld (developed with LLVM 23.1), CMake, Ninja, glslang, Python 3.
- **Your own MetalFX files** from macOS 27.2.1 (build 26B131): the framework's `Resources/` folder
  (`default.metallib`, `emit_*.dat`) and the MetalFX binary extracted from the dyld shared cache. m4r ships none
  of Apple's files and nothing derived from them; everything Apple-specific is read from your copy at build time.

## Quick start

1. Build m4r: `cmake -S . -B build -G Ninja && ninja -C build` (tools and tests) and `cmake --build build-win` for `m4r.dll`
   (MinGW, see [docs/optiscaler.md](docs/optiscaler.md)).
2. Patch and build OptiScaler: `scripts/apply_optiscaler_patches.sh <OptiScaler checkout at fd3a45a>`, then build it as usual
   (or with the clang-cl helper in `optiscaler/`).
3. Generate the plans for your game's resolutions from your own MetalFX files, next to OptiScaler's `dxgi.dll`:
   `scripts/install.py --resources <MetalFX Resources> --binary <MetalFX binary> --game <game folder> --display 2560x1440`.
4. In `OptiScaler.ini`: `Dx12Upscaler=metalfx` for the upscaler, and for frame generation `[FrameGen] Enabled=true`,
   `FGInput=upscaler`, `FGOutput=metalfx` (or pick MetalFX in OptiScaler's menu).

Only D3D12 games running under vkd3d-proton are supported, on RDNA3. Not tried in a real game yet; the tests use a D3D12
harness under Proton ([docs/optiscaler.md](docs/optiscaler.md)).

## Build

```bash
cmake -S . -B build -G Ninja && ninja -C build
# translate one kernel from your metallib (example)
python3 air/m4r_air.py /path/to/MetalFX/Resources/default.metallib copyTexture -o build/k/copyTexture.hsaco --tg 8,8,1
./build/test_kernel build/k
```

## Layout

| Path | Contents |
|---|---|
| `air/` | metallib reader, AIR → AMDGPU translator and AIR built-in library |
| `runtime/` | Vulkan context, kernel loader with Metal's binding model, vkhip (from d4r) |
| `patches/optiscaler/` | MetalFX upscaler and frame generation entries for OptiScaler (`scripts/apply_optiscaler_patches.sh`) |
| `optiscaler/` | helper to build a patched OptiScaler with clang-cl on Linux, for the tests (not a distribution) |
| `tests/` | GPU tests of translated kernels, offline runners, and the D3D12 harness that drives OptiScaler under Proton |
| `tools/` | helpers for reading the MetalFX binary (Mach-O sections, annotated disassembly) |
| `docs/` | how the translation works, what is known about MetalFX's pipelines, and what is verified vs inferred |

## License

GPL-3.0-only (see [LICENSE](LICENSE)). `runtime/vkhip/` comes from d4r under the same license. Apple's MetalFX
files, and anything produced from them (translated kernels, extracted parameters), are not covered by this license
and must not be redistributed.
