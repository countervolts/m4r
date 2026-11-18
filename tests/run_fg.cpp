// Interpolates a fixture sequence with MetalFX's frame interpolation plan (tools/prepare_fg.py).
// usage: run_fg <plan dir> <fixture dir> <frames>
// fixture (tests/scene_fg.py): color-NN.rgba16f (output size), depth-NN.r32f, mv-NN.rg16f (input size, input pixels,
// previous - current). Frame t >= 1 produces <plan dir>/interp-NN.rgba16f, the frame between t-1 and t.
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace m4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static int jsonInt(const std::string& j, const std::string& key, int idx) {
    auto p = j.find("\"" + key + "\"");
    p = j.find('[', p);
    for (int i = 0; i < idx; ++i) p = j.find(',', p + 1);
    return std::stoi(j.substr(p + 1));
}

#pragma pack(push, 1)
struct FGParams {   // FrameGenShaderParams (layout from the kernels' metadata), filled as FrameGenImpl::encodeTo does
    float motionVectorScale[2], depthScale, depthBias, deltaTime, prevDeltaTime, proj33, proj34;
    float clipPrevFromClipCur[16];
    uint32_t haveCameraMatrices, pad2, rejectionThreshold, pad;
    float prevCurrentRatio[2], contentScale[2], normalizeContent[2], normalizePrevContent[2];
    uint16_t contentDimensions[2], outputDimensions[2], outputOffset[2], depthOffset[2], motionOffset[2],
        distortionOffset[2], distortionContentDimensions[2];
    uint32_t tail;
};
#pragma pack(pop)
static_assert(sizeof(FGParams) == 176, "FrameGenShaderParams is 176 bytes");

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <plan dir> <fixture dir> <frames>\n", argv[0]); return 2; }
    std::string dir = argv[1], fx = argv[2];
    int frames = std::stoi(argv[3]);
    std::ifstream sf(dir + "/sizes.json");
    std::string sizes((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
    const int ow = jsonInt(sizes, "output", 0), oh = jsonInt(sizes, "output", 1);
    const int iw = jsonInt(sizes, "input", 0), ih = jsonInt(sizes, "input", 1);
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    int rc = 0;
    {
        Plan plan(rt);
        if (!plan.load(dir)) { std::fprintf(stderr, "plan: %s\n", plan.error.c_str()); return 1; }
        Texture color[2], depth, motion, output;
        vk.createTexture(color[0], VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
        vk.createTexture(color[1], VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
        vk.createTexture(depth, VK_FORMAT_R32_SFLOAT, iw, ih);
        vk.createTexture(motion, VK_FORMAT_R16G16_SFLOAT, iw, ih);
        vk.createTexture(output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
        { VkCommandBuffer cb = vk.begin(); vk.toGeneral(cb, output); vk.submit(cb); }
        plan.bindTexture("depth", &depth); plan.bindTexture("motion", &motion); plan.bindTexture("output", &output);
        VkQueryPool qp;
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP; qci.queryCount = 2;
        vkCreateQueryPool(vk.device, &qci, nullptr, &qp);
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk.physical, &props);
        std::vector<uint8_t> out(size_t(ow) * oh * 8);
        for (int t = 0; t < frames; ++t) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "/color-%02d.rgba16f", t); auto c = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/mv-%02d.rg16f", t); auto m = readAll(fx + nm);
            if (c.size() != size_t(ow) * oh * 8 || d.size() != size_t(iw) * ih * 4 || m.size() != size_t(iw) * ih * 4) {
                std::fprintf(stderr, "frame %d: fixture size mismatch\n", t); rc = 1; break;
            }
            Texture& cur = color[t & 1];
            vk.upload(cur, c.data(), c.size()); vk.upload(depth, d.data(), d.size()); vk.upload(motion, m.data(), m.size());
            plan.bindTexture("color", &cur);
            plan.bindTexture("prevcolor", &color[(t + 1) & 1]);
            FGParams p{};
            p.motionVectorScale[0] = float(ow) / iw; p.motionVectorScale[1] = float(oh) / ih;   // input → output pixels
            p.depthScale = 1.0f; p.depthBias = 0.0f;   // reversed depth
            p.deltaTime = p.prevDeltaTime = 1.0f / 60.0f;
            const float n = 0.1f, f = 1000.0f;
            p.proj33 = n / (n - f); p.proj34 = -n * f / (n - f);
            for (int i = 0; i < 4; ++i) p.clipPrevFromClipCur[i * 5] = 1.0f;
            p.rejectionThreshold = uint32_t(uint64_t(iw) * ih * 80 / 900);
            p.prevCurrentRatio[0] = p.prevCurrentRatio[1] = 1.0f;
            p.contentScale[0] = float(iw) / ow; p.contentScale[1] = float(ih) / oh;
            p.normalizeContent[0] = p.normalizePrevContent[0] = 1.0f / iw;
            p.normalizeContent[1] = p.normalizePrevContent[1] = 1.0f / ih;
            p.contentDimensions[0] = uint16_t(iw); p.contentDimensions[1] = uint16_t(ih);
            p.outputDimensions[0] = uint16_t(ow); p.outputDimensions[1] = uint16_t(oh);
            std::memcpy(plan.buffer("fgparams")->map, &p, sizeof p);
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
            std::printf("frame %d: %.3f ms GPU\n", t, double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6);
            if (t == 0) continue;
            vk.download(output, out.data(), out.size());
            std::snprintf(nm, sizeof nm, "/interp-%02d.rgba16f", t);
            std::ofstream(dir + nm, std::ios::binary).write(reinterpret_cast<char*>(out.data()), out.size());
        }
        if (const char* dump = std::getenv("M4R_DUMP")) {   // raw contents of plan textures after the last frame
            std::string list = dump;
            size_t at = 0;
            while (at < list.size()) {
                size_t e = list.find(',', at);
                std::string n = list.substr(at, e == std::string::npos ? std::string::npos : e - at);
                at = e == std::string::npos ? list.size() : e + 1;
                Texture* tx = plan.texture(n);
                if (!tx) continue;
                std::vector<uint8_t> raw(size_t(tx->width) * tx->height * formatBytes(tx->format));
                vk.download(*tx, raw.data(), raw.size());
                std::ofstream(dir + "/dump-" + n + ".raw", std::ios::binary).write(reinterpret_cast<char*>(raw.data()), raw.size());
                std::printf("dumped %s %ux%u\n", n.c_str(), tx->width, tx->height);
            }
        }
        vkDestroyQueryPool(vk.device, qp, nullptr);
        for (Texture* x : {&color[0], &color[1], &depth, &motion, &output}) vk.destroyTexture(*x);
    }
    rt.destroy();
    vk.destroy();
    return rc;
}
