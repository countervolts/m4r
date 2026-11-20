# DLSS Ray Reconstruction (NGX feature 13) inputs → MetalFX denoised scaler

Cyberpunk 2077 (the Windows build, under Proton) asks for Ray Reconstruction through Streamline/NGX. This is what it hands
over and how each item maps to `MTLFXTemporalDenoisedScaler`. Sources: d4r (`kernels/support/fsr_rr_prepare.comp`,
`tools/d4r_fsr_rr_backend.inl`, `docs/fsr-ray-regeneration.md`, which already convert the same inputs for AMD's FSR Ray
Regeneration) and OptiScaler's `upscalers/dlssd`. Items marked *unverified* are not confirmed for MetalFX.

## Per-frame inputs (NGX parameter keys)

| NGX key | meaning (from d4r) | MetalFX property |
|---|---|---|
| `Color` | one noisy colour at render size (no separate diffuse/specular signals) | `colorTexture` |
| `Depth` | hardware depth if `DLSS.Use.HW.Depth` = 1, else linear view depth; reversed when creation flag `DepthInverted` (0x8) is set | `depthTexture`, `depthReversed`; linear depth needs a conversion pass (d4r builds hardware or signed linear depth from `ViewToClipMatrix`) |
| `MotionVectors` | previous − current, in MV-texture pixels; creation flags `MVLowRes` (0x2), `MVJittered` (0x4) | `motionTexture`, `motionVectorScaleX/Y` from `MV.Scale.X/Y` |
| `DLSS.Input.DiffuseAlbedo` (or `GBuffer.DiffuseAlbedo`) | linear albedo | `diffuseAlbedoTexture` |
| `DLSS.Input.SpecularAlbedo` (or `GBuffer.SpecularAlbedo`) | linear albedo | `specularAlbedoTexture` |
| `GBuffer.Normals` | world-space normals; signed floats, or [0,1] encoded for UNORM formats; roughness in `.w` when `DLSS.Roughness.Mode` = 1 | `normalTexture` (+ a split pass if roughness is packed) |
| `GBuffer.Roughness` | roughness in `.r` when not packed | `roughnessTexture` |
| `DLSSD.SpecularHitDistance` / `DLSSD.SpecularRayDirectionHitDistance` (distance in `.w`) | specular ray hit distance, optional | `specularHitDistanceTexture` |
| `DLSSD.DiffuseHitDistance` / `...RayDirectionHitDistance` | diffuse hit distance, optional | no MetalFX equivalent |
| `DLSSD.ColorBeforeParticles` | colour before particles, optional | possibly `transparencyOverlayTexture` (*unverified* how MetalFX defines the overlay) |
| `DLSSD.OutputAlpha` | alpha output request | not supported (d4r refuses these frames too) |
| `ExposureTexture`, `DLSS.Pre.Exposure`, `DLSS.Exposure.Scale` | exposure | `exposureTexture`, `preExposure` |
| `Jitter.Offset.X/Y` | sub-pixel jitter | `jitterOffsetX/Y` (sign convention as for the upscaler, `[MetalFX] FlipJitter`) |
| `Reset` | history reset | `shouldResetHistory` |
| `WorldToViewMatrix`, `ViewToClipMatrix` | row-major float[16], row-vector convention (v' = v·M) | `worldToViewMatrix`, `viewToClipMatrix`: simd_float4x4 is column-major with column vectors, so transpose |
| `DLSS.Render.Subrect.Dimensions.*`, `DLSS.Input.*.Subrect.Base.*`, `DLSS.Output.Subrect.Base.*` | sub-rectangles of the game's textures | m4r's content/offset parameters (the upscaler already handles colour/depth/motion offsets) |
| `Output` | display-size result | `outputTexture` |
| — | — | `denoiseStrengthMaskTexture`: no DLSS-RR source; leave it disabled |

Creation parameters carry `DLSS.Feature.Create.Flags`, `DLSS.Roughness.Mode`, `DLSS.Use.HW.Depth`, `DLSS.Denoise.Mode`,
`DLSS.Enable.Output.Subrects` and the render/target sizes; d4r resolves per-frame values against the creation values
(a frame may omit creation-only scalars) and OptiScaler's `DLSSDFeature` logs the same ones.

## What Cyberpunk 2077 actually sends (observed 2026-11-19, OptiScaler log, 853x480 -> 2560x1440, path tracing + RR)

| input | provided | DXGI format |
|---|---|---|
| Color | yes | 10 (R16G16B16A16_FLOAT) |
| Depth | yes, hardware depth, inverted (`DLSS.Use.HW.Depth` = 1, creation flag DepthInverted) | 39 (R32_TYPELESS) |
| MotionVectors | yes, render size (`LowResMV`), not jittered; `MV.Scale` = 853, 480 (the render size) | 10 |
| DLSS.Input.DiffuseAlbedo / SpecularAlbedo | yes | 28 (R8G8B8A8_UNORM) |
| GBuffer.Normals | yes, roughness packed in `.w` (`DLSS.Roughness.Mode` = 1), so there is no `GBuffer.Roughness` | 10 |
| DLSSD.SpecularHitDistance | yes | 41 (R32_FLOAT) |
| DLSSD.DiffuseHitDistance | no | |
| DLSSD.ColorBeforeParticles | yes | 10 |
| ExposureTexture | no (`AutoExposure` false, `DLSS.Pre.Exposure` = 1.0) | |
| WorldToViewMatrix / ViewToClipMatrix | yes | |
| Output | 2560x1440 | 10 |

Jitter is in pixels within +-0.5 (e.g. 0.091,-0.369 on the first frame, `Reset` = 1 on frame 0). `DLSS.Denoise.Mode` = 1.
(d4r noted a zero specular hit distance in its Cyberpunk captures; whether this one carries values is not checked.)

## Caveats d4r found with real games

- Normals are assumed to be world space; Alan Wake 2's sky has zero normals with finite far depth (treat as no surface).
- d4r's notes say Cyberpunk path tracing leaves `DLSSD.SpecularHitDistance` zero everywhere and `ColorBeforeParticles`
  unfilled even when bound: do not rely on either until checked against real texel values.
- Albedo energy can exceed 1 (diffuse + specular > 1); d4r takes the excess from the diffuse lobe first before demodulating.
- MetalFX's own input conventions (normal space, depth range, how `transparencyOverlay` and hit distance are used) are
  decided in the unread parts of `encodeToCommandBuffer:`; see `docs/metalfx-denoiser.md` "Not done".

## Getting the game to send it (done, 2026-11-19)

Two things had to change, both in patch 0003: `SuperSamplingDenoising.Available` is reported as 1, and
`NVSDK_NGX_D3D12_GetFeatureRequirements` answers feature 13 as supported. Streamline's `sl.dlss_d` asks for the latter at
startup and, without it, logs "DLSSD feature is not supported" and the game greys the option out. With both, Cyberpunk offers
Ray Reconstruction, creates the feature and evaluates it every frame (the log lines above). The text below is the original analysis.

On an AMD GPU OptiScaler reports `SuperSamplingDenoising.Available = 0` (`NVNGX_Parameter.cpp`, only for non-DLSS-capable
GPUs) and a feature-13 request falls back to FSR 2.1.2 (`FeatureProvider_Dx12.cpp`, case `Upscaler::DLSSD`), so Cyberpunk
does not offer Ray Reconstruction. The wiring needed (patch 0003): report RR available when the m4r denoiser plan exists,
make `FeatureProvider_Dx12` create an `MetalFXDenoiserFeature_Dx12` for `Upscaler::DLSSD`, and check the Streamline path
(`hooks/Streamline_Hooks.cpp` currently does not spoof DLSSD; Cyberpunk ships `sl.dlss_d.dll`). d4r proves the game-side flow:
it ran Cyberpunk 2.31 with RR on this machine's class of hardware through its own NGX shim (`docs/performance.md`).
The first useful build is a capture-only feature: accept feature 13, log the keys above, dump the guide textures per frame
(like `M4R_DUMP_UP` for the upscaler) and output the noisy colour, so the real Cyberpunk inputs can drive the MetalFX
denoiser offline before it exists on the GPU.
