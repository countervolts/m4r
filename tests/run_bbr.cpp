// Upscales a fixture sequence with MetalFX's ML temporal scaler (BBR) plan.
// usage: run_bbr <plan dir> <fixture dir> <frames> [jitter sign +1/-1]
// fixture: input-NN.rgba16f (input size), depth-NN.r32f, mvlo-NN.rg16f (input pixels, previous - current);
// writes <plan dir>/out-NN.rgba16f (output size).
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace m4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double halton(int i, int b) {
    double v = 0, s = 1;
    while (i) { s /= b; v += (i % b) * s; i /= b; }
    return v - 0.5;
}

static int jsonInt(const std::string& j, const std::string& key, int idx) {
    auto p = j.find("\"" + key + "\"");
    p = j.find('[', p);
    for (int i = 0; i < idx; ++i) p = j.find(',', p + 1);
    return std::stoi(j.substr(p + 1));
}

#pragma pack(push, 1)
struct Params {   // BRNetShaderParams (layout from the kernels' metadata)
    uint16_t contentSize[2], maxInputSize[2], outputSize[2], colorOffset[2], motionOffset[2], depthOffset[2],
        reactiveOffset[2], outputOffset[2];
    float jitter[2], motionScale[2], outputResMotionMultiplier[2], normalizeMotionScale[2];
    uint32_t spatialIn[4], spatialOut[4], temporalIn[4], temporalOut[4], history[4];
};
#pragma pack(pop)
static_assert(sizeof(Params) == 144, "BRNetShaderParams is 144 bytes");

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <plan dir> <fixture dir> <frames> [jitter sign]\n", argv[0]); return 2; }
    std::string dir = argv[1], fx = argv[2];
    int frames = std::stoi(argv[3]);
    float jsign = argc > 4 ? std::stof(argv[4]) : 1.0f;
    std::string sizes((std::istreambuf_iterator<char>(*new std::ifstream(dir + "/sizes.json"))), std::istreambuf_iterator<char>());
    const int ow = jsonInt(sizes, "output", 0), oh = jsonInt(sizes, "output", 1);
    const int iw = jsonInt(sizes, "input", 0), ih = jsonInt(sizes, "input", 1);
    const int sw = jsonInt(sizes, "spatial", 0), tw = jsonInt(sizes, "temporal", 0);
    const int hp = 32 * ((2 * ow + 63) / 64);

    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    int rc = 0;
    {
        Plan plan(rt);
        if (!plan.load(dir)) { std::fprintf(stderr, "plan: %s\n", plan.error.c_str()); return 1; }
        Texture color, depth, motion, exposure, output;
        vk.createTexture(color, VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih);
        vk.createTexture(depth, VK_FORMAT_R32_SFLOAT, iw, ih);
        vk.createTexture(motion, VK_FORMAT_R16G16_SFLOAT, iw, ih);
        vk.createTexture(exposure, VK_FORMAT_R32_SFLOAT, 1, 1);
        vk.createTexture(output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
        const float one = 1.0f;
        vk.upload(exposure, &one, 4);
        { VkCommandBuffer cb = vk.begin(); vk.toGeneral(cb, output); vk.submit(cb); }
        plan.bindTexture("color", &color); plan.bindTexture("depth", &depth); plan.bindTexture("motion", &motion);
        plan.bindTexture("exposure", &exposure); plan.bindTexture("output", &output);
        VkQueryPool qp;
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP; qci.queryCount = 2;
        vkCreateQueryPool(vk.device, &qci, nullptr, &qp);
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk.physical, &props);
        std::vector<uint8_t> out(size_t(ow) * oh * 8);
        for (int t = 0; t < frames; ++t) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "/input-%02d.rgba16f", t); auto c = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/mvlo-%02d.rg16f", t); auto m = readAll(fx + nm);
            if (c.size() != size_t(iw) * ih * 8 || d.size() != size_t(iw) * ih * 4 || m.size() != size_t(iw) * ih * 4) {
                std::fprintf(stderr, "frame %d: fixture size mismatch\n", t); rc = 1; break;
            }
            vk.upload(color, c.data(), c.size()); vk.upload(depth, d.data(), d.size()); vk.upload(motion, m.data(), m.size());
            Params p{};
            p.contentSize[0] = p.maxInputSize[0] = uint16_t(iw); p.contentSize[1] = p.maxInputSize[1] = uint16_t(ih);
            p.outputSize[0] = uint16_t(ow); p.outputSize[1] = uint16_t(oh);
            p.jitter[0] = jsign * float(halton(t + 1, 2)); p.jitter[1] = jsign * float(halton(t + 1, 3));
            p.motionScale[0] = p.motionScale[1] = 1.0f;
            p.outputResMotionMultiplier[0] = float(iw) / ow; p.outputResMotionMultiplier[1] = float(ih) / oh;
            p.normalizeMotionScale[0] = 1.0f / iw; p.normalizeMotionScale[1] = 1.0f / ih;
            uint32_t si[4] = {4, uint32_t(4 * sw), 1, 0}, so[4] = {8, uint32_t(8 * sw), 1, 0};
            uint32_t ti[4] = {32, uint32_t(32 * tw), 1, 0}, to[4] = {2, uint32_t(4 * tw), 1, 0};
            uint32_t hs[4] = {1, uint32_t(hp), uint32_t(hp * oh), 0};
            std::memcpy(p.spatialIn, si, 16); std::memcpy(p.spatialOut, so, 16); std::memcpy(p.temporalIn, ti, 16);
            std::memcpy(p.temporalOut, to, 16); std::memcpy(p.history, hs, 16);
            std::memcpy(plan.buffer("params")->map, &p, sizeof p);
            float pre = 1.0f;
            std::memcpy(plan.buffer("preexp")->map, &pre, 4);
            *static_cast<uint8_t*>(plan.buffer("reset")->map) = t == 0;
            float cpf[2] = {float(iw) / ow, float(ih) / oh};
            std::memcpy(plan.buffer("cpf")->map, cpf, 8);
            plan.setParity(t);
            rt.resetTransient();
            VkCommandBuffer cb = vk.begin();
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            if (!plan.record(cb)) { std::fprintf(stderr, "record: %s\n", plan.error.c_str()); rc = 1; break; }
            vk.barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); rc = 1; break; }
            uint64_t ts[2];
            vkGetQueryPoolResults(vk.device, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            vk.download(output, out.data(), out.size());
            std::snprintf(nm, sizeof nm, "/out-%02d.rgba16f", t);
            std::ofstream(dir + nm, std::ios::binary).write(reinterpret_cast<char*>(out.data()), out.size());
            std::printf("frame %d: %.3f ms GPU\n", t, double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6);
        }
        vkDestroyQueryPool(vk.device, qp, nullptr);
        for (Texture* x : {&color, &depth, &motion, &exposure, &output}) vk.destroyTexture(*x);
    }
    rt.destroy();
    vk.destroy();
    return rc;
}
