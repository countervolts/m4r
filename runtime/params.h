// m4r: MetalFX shader parameter blocks (field layouts from the kernels' metadata) and how the host fills them.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cstdint>
#include <cstring>

namespace m4r {

#pragma pack(push, 1)
struct BRNetShaderParams {   // 144 bytes
    uint16_t contentSize[2], maxInputSize[2], outputSize[2], colorOffset[2], motionOffset[2], depthOffset[2],
        reactiveOffset[2], outputOffset[2];
    float jitter[2], motionScale[2], outputResMotionMultiplier[2], normalizeMotionScale[2];
    uint32_t spatialIn[4], spatialOut[4], temporalIn[4], temporalOut[4], history[4];
};
struct FrameGenShaderParams {   // 176 bytes
    float motionVectorScale[2], depthScale, depthBias, deltaTime, prevDeltaTime, proj33, proj34;
    float clipPrevFromClipCur[16];
    uint32_t haveCameraMatrices, pad2, rejectionThreshold, pad;
    float prevCurrentRatio[2], contentScale[2], normalizeContent[2], normalizePrevContent[2];
    uint16_t contentDimensions[2], outputDimensions[2], outputOffset[2], depthOffset[2], motionOffset[2],
        distortionOffset[2], distortionContentDimensions[2];
    uint32_t tail;
};
#pragma pack(pop)
static_assert(sizeof(BRNetShaderParams) == 144, "BRNetShaderParams");
static_assert(sizeof(FrameGenShaderParams) == 176, "FrameGenShaderParams");

// BBRNet_Filter::init / updateBRNetParams; jitter is the offset the frame was rendered with (MetalFX takes its
// negative), motion scale converts the motion texture to input pixels (previous - current)
inline BRNetShaderParams bbrParams(int ow, int oh, int iw, int ih, int cw, int ch, float jx, float jy, float msx, float msy) {
    BRNetShaderParams p{};
    const int sw = ((ow + 7) >> 1) & ~3, tw = ((ow + 15) >> 2) & ~3, hp = 32 * ((2 * ow + 63) / 64);
    p.contentSize[0] = uint16_t(cw); p.contentSize[1] = uint16_t(ch);
    p.maxInputSize[0] = uint16_t(iw); p.maxInputSize[1] = uint16_t(ih);
    p.outputSize[0] = uint16_t(ow); p.outputSize[1] = uint16_t(oh);
    p.jitter[0] = -jx; p.jitter[1] = -jy;
    p.motionScale[0] = msx; p.motionScale[1] = msy;
    p.outputResMotionMultiplier[0] = float(cw) / ow; p.outputResMotionMultiplier[1] = float(ch) / oh;
    p.normalizeMotionScale[0] = msx / iw; p.normalizeMotionScale[1] = msy / ih;
    const uint32_t si[4] = {4, uint32_t(4 * sw), 1, 0}, so[4] = {8, uint32_t(8 * sw), 1, 0};
    const uint32_t ti[4] = {32, uint32_t(32 * tw), 1, 0}, to[4] = {2, uint32_t(4 * tw), 1, 0};
    const uint32_t hs[4] = {1, uint32_t(hp), uint32_t(hp * oh), 0};
    std::memcpy(p.spatialIn, si, 16); std::memcpy(p.spatialOut, so, 16); std::memcpy(p.temporalIn, ti, 16);
    std::memcpy(p.temporalOut, to, 16); std::memcpy(p.history, hs, 16);
    return p;
}

// FrameGenImpl::encodeTo (no camera matrices, no distortion field)
inline FrameGenShaderParams fgParams(int ow, int oh, int iw, int ih, float msx, float msy, bool reversed, float dt,
                                     float prevDt, float n, float f) {
    FrameGenShaderParams p{};
    p.motionVectorScale[0] = msx * float(ow) / iw; p.motionVectorScale[1] = msy * float(oh) / ih;
    p.depthScale = reversed ? 1.0f : -1.0f; p.depthBias = reversed ? 0.0f : 1.0f;
    p.deltaTime = dt; p.prevDeltaTime = prevDt;
    p.proj33 = n / (n - f); p.proj34 = -n * f / (n - f);
    for (int i = 0; i < 4; ++i) p.clipPrevFromClipCur[i * 5] = 1.0f;
    p.rejectionThreshold = uint32_t(uint64_t(iw) * ih * 80 / 900);
    p.prevCurrentRatio[0] = p.prevCurrentRatio[1] = 1.0f;
    p.contentScale[0] = float(iw) / ow; p.contentScale[1] = float(ih) / oh;
    p.normalizeContent[0] = p.normalizePrevContent[0] = 1.0f / iw;
    p.normalizeContent[1] = p.normalizePrevContent[1] = 1.0f / ih;
    p.contentDimensions[0] = uint16_t(iw); p.contentDimensions[1] = uint16_t(ih);
    p.outputDimensions[0] = uint16_t(ow); p.outputDimensions[1] = uint16_t(oh);
    return p;
}

} // namespace m4r
