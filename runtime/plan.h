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
    // Records every pass into cb (one barrier between passes).
    bool record(VkCommandBuffer cb);
    size_t passes() const { return passes_.size(); }
    const std::map<std::string, Kernel*>& kernels() const { return kernels_; }
    std::string error;

private:
    struct Bind { char kind; uint32_t index; std::string name; std::vector<uint8_t> bytes; };
    struct Pass { Kernel* kernel = nullptr; bool groups = false; uint32_t grid[3] = {1, 1, 1}; std::vector<Bind> binds; std::string label; };
    Runtime& rt_;
    std::map<std::string, Buffer> buffers_;
    std::map<std::string, Texture> textures_;
    std::map<std::string, Kernel*> kernels_;
    std::map<std::string, const Texture*> extTex_;
    std::map<std::string, VkDeviceAddress> extBuf_;
    std::vector<Pass> passes_;
};

VkFormat formatByName(const std::string& n);

} // namespace m4r
