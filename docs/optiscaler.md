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

Next to the game executable (for the D3D12 games OptiScaler supports, running under Proton) you need the patched OptiScaler
(`dxgi.dll` and its ini) and, from m4r:

```
m4r\m4r.dll
m4r\plans\bbr-<display w>x<display h>-<render w>x<render h>        upscaler plan, reversed depth
m4r\plans\bbr-<display w>x<display h>-<render w>x<render h>-fwd    upscaler plan, forward depth
m4r\plans\fg-<display w>x<display h>-<render w>x<render h>         frame generation plan
```

and in `OptiScaler.ini`: `Dx12Upscaler=metalfx` and/or `[FrameGen] Enabled=true`, `FGInput=upscaler`, `FGOutput=metalfx`.

`scripts/install.py` makes all of that from your own copy of MetalFX (the framework's `Resources` folder and the MetalFX binary
extracted from the dyld shared cache) in about two seconds per plan:

```bash
scripts/install.py --resources .../MetalFX.framework/Versions/A/Resources --binary .../MetalFX --game /path/to/game \
    --display 2560x1440 [--modes quality,balanced,performance,ultra] [--input 1706x960] [--depth both] [--no-fg]
```

It works out each quality mode's render size the way OptiScaler does and writes a plan per display and render size pair; sizes a
game chooses on its own (dynamic resolution) are added with `--input`. A plan is needed for every pair that will be used: without
one, `m4r_upscale_init` fails and the feature reports it in OptiScaler's log (the error names the missing plan). The plans contain
Apple's translated kernels and weights and must not be shared.

## What happens

**Upscaler.** At `EvaluateFeature`, OptiScaler gives the feature the game's D3D12 resources. They are turned into
`VkImage`s with `ID3D12DXVKInteropDevice1::GetVulkanResourceInfo1` and the layout for their D3D12 state, m4r's 27 passes
are recorded into the game's own command list through `BeginVkCommandBufferInterop`, and nothing is waited for. The NGX
jitter is passed as it is; m4r negates it internally (MetalFX's convention). m4r expects the sample offset, i.e. the frame
was rendered at pixel + 0.5 + jitter. `[MetalFX] FlipJitter` is for games that report the opposite sign; whether one does can
be measured instead of judged by eye: in static pixels of two consecutive captured inputs (`M4R_DUMP_UP`) the content shifts by
the jitter difference (a least-squares gradient fit gives the shift). Ghost of Tsushima measured -1x the passed difference with
FlipJitter=true in every frame, so it needs the default (false); with the wrong sign the history is merged at mirrored
sub-pixel positions (softer each frame, shimmering).

Exposure: when the game gives no exposure texture (DLSS AutoExposure, as Ghost of Tsushima), m4r runs MetalFX's own auto
exposure (`luma_log_sum`, `luma_log_sum_to_exposure`, as MetalFX does for a descriptor with auto exposure enabled) instead of a
fixed 1.0; `M4R_NO_AUTOEXPOSURE=1` restores 1.0.

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
| Upscaler through OptiScaler under Proton | works with the test harness: 640x360 → 1280x720, 28.27 dB against a supersampled reference (bilinear 21.9 dB), about 2.7 ms per frame including submit and wait. **Not tried in a game yet.** |
| Frame generation output | **works in the test harness** (swapchain, 20 frames, 1280x720, OptiFG with the Upscaler input): from the first active frame every present becomes two; each real frame is put back bit-exactly; generated frames equal a native replay of the same parameters (8 of 9 bit for bit, one frame differs in 15 values); about 24.5–25.1 dB against the true in-between frame where blending gives 21.7 dB. **Not tried in a game yet.** |
| Menu entries | added; not seen on screen yet |

Known limits: display-resolution motion vectors (the upscaler logs a warning, frame generation skips the frame),
one plan per resolution pair, vkd3d-proton only, RDNA3 only (gfx1101 tested), frame generation interpolates the final image
including the HUD (no hudless input), no dynamic resolution.

The pipeline is deterministic: repeated runs give bit-identical frames (`tests/check_determinism.sh`).

## Frame generation behaviour seen in the test

- The first generated frame after a reset is a copy of the current frame: there is no history yet (the offline runner shows the
  same). Other frames can be copies too: the histogram analysis compares nine 256-bin histograms of consecutive frames and sets
  a bit for each where the sum of absolute differences reaches `w*h*80/900` of the input; if any bit is set the warp stage copies
  the current frame (the sketch is in docs/metalfx-framegen.md). On the test scene one frame in nine (18) trips it. This is
  Apple's logic, executed as written, and a harmless fallback.
- **Open, small:** in one frame (12) the generated frame differs from the native replay in 6 to 15 values (a few pixels around
  y=385..388, up to 0.18 in value, 73 dB PSNR), consistently in the harness run and consistently in the replay, never in the
  other frames. What I ruled out: the inputs (depth, motion vectors and the back buffer are bit-identical to the fixture, dumped
  inside m4r with `M4R_DUMP_FG`), the kernel parameter bytes (identical, logged with `M4R_LOG`), uninitialised video memory
  (junk-filled before the replay), GPU timing and occupancy (the replay is identical under load and with large LDS
  reservations), the upscaler running before it, and the queue family. A race between writers in a motion-vector splat pass is
  the most likely explanation, but I have not shown it.

## Testing

Frame generation, through OptiScaler under Proton, then compared with a native replay of the same frames and parameters:

```bash
M4R_FG_PLAN=build/fg-1280x720 tests/run_optiscaler.sh "$PROTON_DIR" build-opti/OptiScaler.dll build-win/m4r.dll \
    build/bbr-1280x720 build/fixtures/motion20 20          # build/fixtures/motion20: M4R_N=20 M4R_OBJ=mv M4R_MOTION_DIR=motion20 python3 tests/scene_motion.py
MID_DIR=build/fixtures/fg20 BBR_PLAN=build/bbr-1280x720 \
    tests/check_framegen.sh build-harness/app build/fixtures/motion20 build/fg-1280x720 20   # fg20: M4R_FG_DIR=fg20 M4R_N=20 M4R_OBJ=mv python3 tests/scene_fg.py
```

OptiScaler pauses frame generation for 10 frames after it activates, so the run needs more than 14 frames. The harness paces
itself at 60 fps and keeps its frames in memory until the end so that OptiScaler sees plausible frame times.

Upscaler only:

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
