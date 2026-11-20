// m4r: pass plan executor (see plan.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <set>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace m4r {

VkFormat formatByName(const std::string& n) {
    static const std::map<std::string, VkFormat> f = {
        {"R8_UNORM", VK_FORMAT_R8_UNORM}, {"R8G8_UNORM", VK_FORMAT_R8G8_UNORM}, {"R16_SFLOAT", VK_FORMAT_R16_SFLOAT},
        {"R16G16_SFLOAT", VK_FORMAT_R16G16_SFLOAT}, {"R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"R32_SFLOAT", VK_FORMAT_R32_SFLOAT}, {"R32_UINT", VK_FORMAT_R32_UINT}, {"R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM},
        {"R16G16_UINT", VK_FORMAT_R16G16_UINT}, {"R8_UINT", VK_FORMAT_R8_UINT}, {"R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT}};
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
        if (w == "gpu") {   // gpu gfx1101: the kernels below are code for that GPU only
            ls >> gpu_;
            const std::string& dev = rt_.vk().gpu;
            if (!dev.empty() && gpu_ != dev)
                return fail("this plan was built for " + gpu_ + " but the GPU is " + dev + "; reinstall the plans for it (install.py "
                            "--gpu " + dev + ")");
            continue;
        }
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
            textures_[name] = t;
            uninitialised_.push_back(name);   // transitioned and cleared in the first recorded frame
        } else if (w == "kernel") {
            std::string name, path, opt;
            ls >> name >> path;
            uint32_t lds = 0;
            while (ls >> opt) if (opt.rfind("lds=", 0) == 0) lds = uint32_t(std::stoul(opt.substr(4)));
            Kernel* k = rt_.load(dir + "/" + path, lds);
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
        } else if (w == "view") {      // view NAME BUFFER OFFSET: a buffer address at an offset
            std::string name, base;
            VkDeviceSize off;
            ls >> name >> base >> off;
            views_[name] = {base, off};
        } else if (w == "only") {      // only FLAG / only !FLAG: the pass runs only while the flag is set / clear
            if (!cur) return fail("only outside a pass");
            std::string fl;
            ls >> fl;
            cur->onlyNot = !fl.empty() && fl[0] == '!';
            cur->only = cur->onlyNot ? fl.substr(1) : fl;
            if (cur->only.empty()) return fail("only without a flag");
        } else if (w == "buf" || w == "tex" || w == "bytes" || w == "tgmem") {
            if (!cur) return fail("binding outside a pass");
            Bind b;
            b.kind = w == "bytes" ? 'y' : w == "tgmem" ? 'g' : w[0];
            ls >> b.index >> b.name;
            if (w == "bytes") b.bytes = hexBytes(b.name);
            cur->binds.push_back(b);
        } else {
            return fail("unknown directive " + w);
        }
    }
    // Metal bindings stay in place from one dispatch to the next (MetalFX binds the parameter buffer once for several
    // kernels). Give every pass the bindings its kernel's arguments need, so that any range of passes can be recorded alone.
    std::map<int, Bind> buffers, textures;
    for (auto& p : passes_) {
        for (auto& b : p.binds) {
            if (b.kind == 'b' || b.kind == 'y') buffers[int(b.index)] = b;
            else if (b.kind == 't') textures[int(b.index)] = b;
        }
        for (const Arg& a : p.kernel->args) {
            if (a.location < 0) continue;
            const bool isBuffer = a.kind == Arg::Buffer, isTexture = a.kind == Arg::Texture;
            if (!isBuffer && !isTexture) continue;
            bool bound = false;
            for (auto& b : p.binds)
                if (int(b.index) == a.location && (isBuffer ? (b.kind == 'b' || b.kind == 'y') : b.kind == 't')) bound = true;
            if (bound) continue;
            auto& have = isBuffer ? buffers : textures;
            if (auto it = have.find(a.location); it != have.end()) p.binds.push_back(it->second);
        }
    }
    return true;
}

bool Plan::record(VkCommandBuffer cb, size_t first, size_t count) {
    for (auto& n : uninitialised_) {
        Texture& t = textures_[n];
        rt_.vk().toGeneral(cb, t);
        VkClearColorValue zero{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
    }
    uninitialised_.clear();
    Encoder enc(rt_, cb);
    // Passes run in dependency levels: a pass's level is one more than the highest level of an earlier pass whose output it
    // reads or writes, or whose input it overwrites; within a level passes are independent and run without barriers between
    // them (one barrier per level). Every dependency is kept, so the results equal the plan's order. Kernels without a
    // manifest (GLSL passes) count as writing everything they bind. M4R_ALL_BARRIERS=1: plan order, a barrier before each pass.
    const size_t last = first >= passes_.size() ? passes_.size() : (count > passes_.size() - first ? passes_.size() : first + count);
    std::vector<std::vector<std::pair<std::string, bool>>> uses(last - first);   // resource, written
    for (size_t pi = first; pi < last; ++pi) {
        Pass& p = passes_[pi];
        if (!active(p)) continue;   // switched off: no uses, recorded below as nothing
        for (auto& bb : p.binds) {
            if (bb.kind != 'b' && bb.kind != 't') continue;
            std::string nm = pick(bb.name);
            if (bb.kind == 'b') { auto v = views_.find(nm); if (v != views_.end()) nm = v->second.first; }
            bool known = false, wr = true;
            for (const auto& a : p.kernel->args)
                if (a.location == int(bb.index) && ((bb.kind == 'b' && a.kind == Arg::Buffer) || (bb.kind == 't' && a.kind == Arg::Texture))) {
                    known = true; wr = !a.readonly;
                }
            if (!known && !p.kernel->args.empty()) continue;   // bound but not an argument of this kernel
            uses[pi - first].emplace_back((bb.kind == 'b' ? "b:" : "t:") + nm, wr);
        }
    }
    const bool serial = std::getenv("M4R_ALL_BARRIERS") != nullptr;
    std::vector<int> level(last - first, 0);
    for (size_t i = 0; i < level.size(); ++i) {
        if (serial) { level[i] = int(i); continue; }
        for (size_t j = 0; j < i; ++j) {
            bool dep = false;
            for (auto& [a, aw] : uses[i])
                for (auto& [b, bw] : uses[j])
                    if (a == b && (aw || bw)) dep = true;
            if (dep) level[i] = std::max(level[i], level[j] + 1);
        }
    }
    std::vector<size_t> order(level.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return level[a] < level[b]; });
    int prevLevel = -1;
    for (size_t oi : order) {
        const size_t pi = first + oi;
        Pass& p = passes_[pi];
        if (!active(p)) continue;
        enc.setKernel(p.kernel);
        enc.setBarrier(level[oi] != prevLevel);
        prevLevel = level[oi];
        for (auto& bb : p.binds) {
            Bind b = bb;
            if (b.kind != 'y') b.name = pick(b.name);
            if (b.kind == 'b') {
                auto e = extBuf_.find(b.name);
                if (e != extBuf_.end()) { enc.setBuffer(e->second, b.index); continue; }
                auto v = views_.find(b.name);
                Buffer* buf = buffer(v != views_.end() ? v->second.first : b.name);
                if (!buf) { error = p.label + ": no buffer " + b.name; return false; }
                enc.setBuffer(*buf, b.index, v != views_.end() ? v->second.second : 0);
            } else if (b.kind == 'g') {
                enc.setThreadgroupMemory(uint32_t(std::stoul(b.name)), b.index);
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
