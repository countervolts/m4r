// m4r: executes a prepared pass plan (tools/prepare_*.py): resources, translated kernels and their bindings.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "kernel.h"
#include <map>
#include <string>
#include <vector>

namespace m4r {

class Plan {
public:
    explicit Plan(Runtime& rt) : rt_(rt) {}
    ~Plan();
    bool load(const std::string& dir);        // <dir>/plan.txt
    // Resources owned by the caller (game textures, per-frame parameters) override plan resources by name.
    void bindTexture(const std::string& name, const Texture* t) { extTex_[name] = t; }
    void bindBuffer(const std::string& name, VkDeviceAddress a) { extBuf_[name] = a; }
    Buffer* buffer(const std::string& name);
    Texture* texture(const std::string& name);
    // Records the passes [first, first + count) into cb (one barrier between passes); all of them by default.
    bool record(VkCommandBuffer cb, size_t first = 0, size_t count = size_t(-1));
    // ping-pong resources are written "a|b" in the plan: a on even, b on odd parity (MetalFX flips once per frame)
    void setParity(int p) { parity_ = p & 1; }
    // passes marked `only FLAG` (or `only !FLAG`) in the plan run only while the flag is set (clear); flags start clear
    void setFlag(const std::string& name, bool on) { flags_[name] = on; }
    size_t passes() const { return passes_.size(); }
    const std::string& passLabel(size_t i) const { return passes_[i].label; }
    const std::map<std::string, Buffer>& buffers() const { return buffers_; }
    const std::map<std::string, Texture>& textures() const { return textures_; }
    const std::map<std::string, Kernel*>& kernels() const { return kernels_; }
    std::string error;

private:
    struct Bind { char kind; uint32_t index; std::string name; std::vector<uint8_t> bytes; };
    struct Pass { Kernel* kernel = nullptr; bool groups = false; uint32_t grid[3] = {1, 1, 1}; std::vector<Bind> binds; std::string label;
                  std::string only; bool onlyNot = false; };
    Runtime& rt_;
    std::map<std::string, Buffer> buffers_;
    std::map<std::string, Texture> textures_;
    std::map<std::string, Kernel*> kernels_;
    std::map<std::string, const Texture*> extTex_;
    std::map<std::string, VkDeviceAddress> extBuf_;
    std::vector<Pass> passes_;
    std::map<std::string, std::pair<std::string, VkDeviceSize>> views_;
    int parity_ = 0;
    std::map<std::string, bool> flags_;
    bool active(const Pass& p) const {
        if (p.only.empty()) return true;
        auto f = flags_.find(p.only);
        return (f != flags_.end() && f->second) != p.onlyNot;
    }
    std::vector<std::string> uninitialised_;
    std::string pick(const std::string& n) const {
        auto bar = n.find('|');
        return bar == std::string::npos ? n : (parity_ ? n.substr(bar + 1) : n.substr(0, bar));
    }
};

VkFormat formatByName(const std::string& n);

} // namespace m4r
