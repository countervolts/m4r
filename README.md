# m4r (metalfx 4 radeon)

want dlss on radeon? check out [d4r](https://github.com/countervolts/d4r)! xess xmx? [x4r](https://github.com/countervolts/x4r)!

m4r runs Apple's **MetalFX** upscaler (the ML temporal scaler, "BBR") and MetalFX frame
interpolation in Windows games on AMD Radeon GPUs under Linux and Proton. The game asks for DLSS/FSR/XeSS as usual;
patches for [OptiScaler](https://github.com/optiscaler/OptiScaler) add **MetalFX** to its upscaler menu and **MetalFX FG**

m4r runs Apple's own kernels and network weights, as d4r does with NVIDIA's and x4r with Intel's: the Metal
compute kernels in your copy of MetalFX's `default.metallib` are translated from their AIR (LLVM) bitcode to
AMDGPU code objects and run inside the game's Vulkan command stream through d4r's **vkhip** loader. No network
is re-trained, approximated or replaced.

> **proof of concept.** Not affiliated with Apple or AMD.

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

## License

GPL-3.0-only (see [LICENSE](LICENSE)). `runtime/vkhip/` comes from d4r under the same license. Apple's MetalFX
files, and anything produced from them (translated kernels, extracted parameters), are not covered by this license
and must not be redistributed.
