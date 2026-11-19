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

## Performance

GPU time per frame on an RX 7700 XT (RDNA3, gfx1101):

| | 2560x1440 output |
|---|---|
| Upscaler, Performance (1280x720 in) | **1.98 ms** |
| Upscaler, Ultra Performance (856x480 in) | 1.96 ms |
| Upscaler with line locking (`install.py --line-locking`) | 2.06 ms |
| Frame generation, per generated frame | **1.55 ms** |

At 1280x720 output (640x360 in) the upscaler takes 0.64 ms.

Every optimisation is bit-identical: the output matches the first, unoptimised translation exactly. Since then the upscaler
went from 6.7 to 1.98 ms and frame generation from 2.9 to 1.55 ms. They are measured offline with the captured game frames
in `build/run_bbr` / `build/run_fg`; [docs/mxu.md](docs/mxu.md) and [docs/air-translation.md](docs/air-translation.md)
describe the main techniques.

RDNA2 has no matrix instructions, so its network runs on software matrix products. On the RX 7700 XT that code path takes
4.2 ms for the 1440p upscaler. It has not been timed on RDNA2 or RDNA4 hardware yet.

## Quick start

1. Build m4r: `cmake -S . -B build -G Ninja && ninja -C build` (tools and tests) and `cmake --build build-win` for `m4r.dll`
   (MinGW, see [docs/optiscaler.md](docs/optiscaler.md)).
2. Patch and build OptiScaler: `scripts/apply_optiscaler_patches.sh <OptiScaler checkout at fd3a45a>`, then build it as usual
   (or with the clang-cl helper in `optiscaler/`).
3. Generate the plans for your game's resolutions from your own MetalFX files, next to OptiScaler's `dxgi.dll`:
   `scripts/install.py --resources <MetalFX Resources> --binary <MetalFX binary> --game <game folder> --display 2560x1440`.
4. In `OptiScaler.ini`: `Dx12Upscaler=metalfx` for the upscaler, and for frame generation `[FrameGen] Enabled=true`,
   `FGInput=upscaler`, `FGOutput=metalfx` (or pick MetalFX in OptiScaler's menu).

Only D3D12 games running under vkd3d-proton are supported, on RDNA2/3/4 (plans are built for your GPU; `install.py` detects it). Not tried in a real game yet; the tests use a D3D12
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
