// m4r: pass plan executor (see plan.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <fstream>
#include <sstream>

namespace m4r {

VkFormat formatByName(const std::string& n) {
    static const std::map<std::string, VkFormat> f = {
        {"R8_UNORM", VK_FORMAT_R8_UNORM}, {"R8G8_UNORM", VK_FORMAT_R8G8_UNORM}, {"R16_SFLOAT", VK_FORMAT_R16_SFLOAT},
        {"R16G16_SFLOAT", VK_FORMAT_R16G16_SFLOAT}, {"R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"R32_SFLOAT", VK_FORMAT_R32_SFLOAT}, {"R32_UINT", VK_FORMAT_R32_UINT}, {"R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM},
        {"R16G16_UINT", VK_FORMAT_R16G16_UINT}, {"R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT}};
    auto it = f.find(n);
    return it == f.end() ? VK_FORMAT_UNDEFINED : it->second;
}

Plan::~Plan() {
    for (auto& [n, b] : buffers_) rt_.vk().destroyBuffer(b);
    for (auto& [n, t] : textures_) rt_.vk().destroyTexture(t);
}

Buffer* Plan::buffer(const std::string& name) {
    auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : &it->second;
}

Texture* Plan::texture(const std::string& name) {
    auto it = textures_.find(name);
    return it == textures_.end() ? nullptr : &it->second;
}

static std::vector<uint8_t> hexBytes(const std::string& h) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(uint8_t(std::stoul(h.substr(i, 2), nullptr, 16)));
    return out;
}

bool Plan::load(const std::string& dir) {
    std::ifstream f(dir + "/plan.txt");
    if (!f) { error = dir + "/plan.txt missing"; return false; }
    Context& vk = rt_.vk();
    std::string line;
    Pass* cur = nullptr;
    int ln = 0;
    while (std::getline(f, line)) {
        ++ln;
        std::istringstream ls(line);
        std::string w;
        if (!(ls >> w) || w[0] == '#') continue;
        auto fail = [&](const std::string& why) { error = "plan.txt:" + std::to_string(ln) + ": " + why; return false; };
        if (w == "buffer") {
            std::string name, src, path;
            size_t size;
            ls >> name >> size >> src >> path;
            Buffer b;
            if (!vk.createBuffer(b, size, true)) return fail("buffer " + name);
            std::memset(b.map, 0, size);
            if (src == "file") {
                std::ifstream bf(dir + "/" + path, std::ios::binary);
                bf.read(static_cast<char*>(b.map), std::streamsize(size));
                if (bf.gcount() != std::streamsize(size)) return fail("short file " + path);
            }
            buffers_[name] = b;
        } else if (w == "texture") {
            std::string name, fmt;
            uint32_t tw, th;
            ls >> name >> fmt >> tw >> th;
            Texture t;
            if (!vk.createTexture(t, formatByName(fmt), tw, th)) return fail("texture " + name);
            VkCommandBuffer cb = vk.begin();
            vk.toGeneral(cb, t);
            VkClearColorValue zero{};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
            if (!vk.submit(cb)) return fail("clear " + name);
            textures_[name] = t;
        } else if (w == "kernel") {
            std::string name, path;
            ls >> name >> path;
            Kernel* k = rt_.load(dir + "/" + path);
            if (!k) return fail(rt_.error);
            kernels_[name] = k;
        } else if (w == "glsl") {      // glsl NAME path.spv tx ty tz
            std::string name, path;
            uint32_t tx, ty, tz;
            ls >> name >> path >> tx >> ty >> tz;
            Kernel* k = rt_.loadGlsl(dir + "/" + path, tx, ty, tz);
            if (!k) return fail(rt_.error);
            kernels_[name] = k;
        } else if (w == "pass") {
            Pass p;
            std::string kname, mode;
            ls >> kname >> mode >> p.grid[0] >> p.grid[1] >> p.grid[2];
            auto it = kernels_.find(kname);
            if (it == kernels_.end()) return fail("unknown kernel " + kname);
            p.kernel = it->second; p.groups = mode == "groups"; p.label = kname;
            passes_.push_back(p);
            cur = &passes_.back();
        } else if (w == "buf" || w == "tex" || w == "bytes") {
            if (!cur) return fail("binding outside a pass");
            Bind b;
            b.kind = w == "bytes" ? 'y' : w[0];
            ls >> b.index >> b.name;
            if (w == "bytes") b.bytes = hexBytes(b.name);
            cur->binds.push_back(b);
        } else {
            return fail("unknown directive " + w);
        }
    }
    return true;
}

bool Plan::record(VkCommandBuffer cb) {
    Encoder enc(rt_, cb);
    for (auto& p : passes_) {
        enc.setKernel(p.kernel);
        for (auto& bb : p.binds) {
            Bind b = bb;
            if (b.kind != 'y') b.name = pick(b.name);
            if (b.kind == 'b') {
                auto e = extBuf_.find(b.name);
                if (e != extBuf_.end()) { enc.setBuffer(e->second, b.index); continue; }
                Buffer* buf = buffer(b.name);
                if (!buf) { error = p.label + ": no buffer " + b.name; return false; }
                enc.setBuffer(*buf, b.index);
            } else if (b.kind == 't') {
                auto e = extTex_.find(b.name);
                const Texture* t = e != extTex_.end() ? e->second : texture(b.name);
                if (!t && b.name != "nil") { error = p.label + ": no texture " + b.name; return false; }
                enc.setTexture(t, b.index);
            } else {
                enc.setBytes(b.bytes.data(), b.bytes.size(), b.index);
            }
        }
        bool ok = p.groups ? enc.dispatchThreadgroups(p.grid[0], p.grid[1], p.grid[2])
                           : enc.dispatchThreads(p.grid[0], p.grid[1], p.grid[2]);
        if (!ok) { error = p.label + ": " + enc.error; return false; }
    }
    return true;
}

} // namespace m4r
