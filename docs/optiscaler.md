# MetalFX in OptiScaler

m4r does not ship an OptiScaler build. It ships patches (`patches/optiscaler/`) against upstream
[OptiScaler](https://github.com/optiscaler/OptiScaler) that add

- **MetalFX** to the upscaler dropdown (`Dx12Upscaler=metalfx`), and
- **MetalFX FG** to the frame generation output list (`FGOutput=metalfx`).

Both hand the work to `m4r\m4r.dll`, which runs Apple's kernels and weights on the game's Vulkan device (vkd3d-proton).
Nothing else in OptiScaler changes: the patches only add things. Without `m4r\m4r.dll` the upscaler entry is not listed,
and selecting the frame generation output shows an error notification and switches frame generation off.

## Patches

Made against OptiScaler commit `fd3a45a` ("Display asi's version", the 10.0.0-pre1 line). Apply them in order to a clean checkout:

```bash
scripts/apply_optiscaler_patches.sh /path/to/OptiScaler        # prints what failed and takes back what it applied
```

| Patch | Adds |
|---|---|
| `0001-metalfx-upscaler.patch` | `Upscaler::MetalFX` (ini code `metalfx`), `MetalFXFeatureDx12` (an NGX D3D12 feature, built on vkd3d-proton's Vulkan interop), the dropdown entry, `[MetalFX] FlipJitter`, the loader for `m4r\m4r.dll`, a copy of `m4r.h`, and `OptiScaler.ini` documentation |
| `0002-metalfx-frame-generation.patch` | `FGOutput::MetalFX` (`MetalFXFG_Dx12`), the "MetalFX FG" entry and a pacing checkbox in the frame generation menu, `[MetalFX] FGPacing`, two `State` fields that carry the game's present arguments |

`m4r.h` inside patch 0001 is a copy of `runtime/m4r.h`; keep them equal when the API changes.

`optiscaler/clang-cl.patch` is **not** part of the series. It holds five small changes that only matter when building
OptiScaler with clang-cl on Linux (`scripts/apply_optiscaler_patches.sh <dir> --clang-cl`), which is how m4r's tests
build it. Upstream builds with MSVC and needs none of it.

## Installing

Next to the game executable (for the D3D12 games OptiScaler supports, running under Proton):

```
dxgi.dll              OptiScaler (patched), or any OptiScaler setup you already use
OptiScaler.ini        Dx12Upscaler=metalfx   and/or   [FrameGen] FGOutput=metalfx
m4r\m4r.dll
m4r\plans\bbr-<out w>x<out h>-<in w>x<in h>   one per upscaling resolution pair in use
m4r\plans\fg-<display w>x<display h>-<in w>x<in h>   one per frame generation resolution pair
```

Plans are generated on your machine from your own copy of MetalFX (`tools/prepare_bbr.py`, `tools/prepare_fg.py`); they
contain Apple's translated kernels and weights and must not be shared. An installer that generates the common
resolutions is not written yet.

## What happens

**Upscaler.** At `EvaluateFeature`, OptiScaler gives the feature the game's D3D12 resources. They are turned into
`VkImage`s with `ID3D12DXVKInteropDevice1::GetVulkanResourceInfo1` and the layout for their D3D12 state, m4r's 27 passes
are recorded into the game's own command list through `BeginVkCommandBufferInterop`, and nothing is waited for. The NGX
jitter is passed as it is; m4r negates it internally (MetalFX's convention).

**Frame generation.** OptiScaler's OptiFG (`FGInput=upscaler`) keeps a copy of the frame's depth and motion vectors.
`MetalFXFG_Dx12` wraps the game's swapchain (the same pass-through wrapper the reprojection output uses). At present:

1. a command list calls `m4r_framegen_capture`, which keeps a copy of the finished backbuffer and, from the second frame on,
   interpolates between the previous and the current frame into the backbuffer;
2. that generated frame is presented on the real swapchain;
3. `m4r_framegen_restore` writes the real frame into the next backbuffer, and OptiScaler presents it as usual.

With V-Sync off, the real frame is held back by half the game's frame time (`FGPacing`) so the two frames are not
shown back to back. How well that paces is a guess; it is not a replacement for a presenter thread.

## Status

| | |
|---|---|
| Upscaler through OptiScaler under Proton | works with the test harness: 640x360 → 1280x720, 28.27 dB against a supersampled reference (bilinear 21.9 dB), about 6.3 ms per frame including submit and wait. **Not tried in a game yet.** |
| Frame generation output | compiles, links and is selectable; **not exercised end to end yet** (the harness has no swapchain mode) |
| Menu entries | added; not seen on screen yet |

Known limits: display-resolution motion vectors (the upscaler logs a warning, frame generation skips the frame),
one plan per resolution pair, vkd3d-proton only, RDNA3 only (gfx1101 tested), frame generation interpolates the final image
including the HUD (no hudless input), no dynamic resolution.

The pipeline is deterministic: repeated runs give bit-identical frames (`tests/check_determinism.sh`).

## Testing

```bash
tests/build_harness.sh                               # clang-cl + xwin -> build-harness/d3d12_harness.exe
cmake --build build-win                              # m4r.dll (MinGW)
scripts/apply_optiscaler_patches.sh external/optiscaler --clang-cl     # once, on a clean pinned checkout
cmake -S optiscaler -B build-opti -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/clang-cl-xwin.cmake && cmake --build build-opti
tests/run_optiscaler.sh "$PROTON_DIR" build-opti/OptiScaler.dll build-win/m4r.dll build/bbr-1280x720 build/fixtures/motion 8
python3 tests/compare.py build-harness/app/out build/fixtures/motion 8 1280 720 640 360
```

The harness calls OptiScaler's NGX D3D12 exports the way a game would and logs to `harness.log` (Proton drops the stdout
of console programs).
