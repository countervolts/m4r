// From d4r (github.com/countervolts/d4r, engine/vkhip), GPL-3.0-only.
// vkhip: see vkhip.h.
#include "vkhip.h"
#include "donor_spv.h"
#include <algorithm>
#include <bit>
#include <cstdio>

namespace vkhip {
namespace {

template <class T> T rd(const std::vector<uint8_t>& b, size_t at) {
    T v{};
    if (at + sizeof(T) <= b.size()) std::memcpy(&v, b.data() + at, sizeof(T));
    return v;
}
template <class T> void wr(std::vector<uint8_t>& b, size_t at, T v) { std::memcpy(b.data() + at, &v, sizeof(T)); }

struct Section { std::string name; uint32_t type = 0; uint64_t flags = 0, addr = 0, offset = 0, size = 0; uint32_t link = 0; };

constexpr uint32_t SHT_NOBITS = 8, SHT_SYMTAB = 2, SHT_DYNSYM = 11;
constexpr uint64_t SHF_WRITE = 1, SHF_ALLOC = 2;

// Encodings of the instructions vkhip reads or writes, per generation (checked with llvm-mc). gfx12 keeps gfx11's;
// gfx10.3 numbers its SOP1/SOPP opcodes differently.
struct Isa { uint32_t getpc, movLit, branch, movB64; };   // movB64: s_mov_b64 s[0:1], s[0:1] (| sdst << 16 | ssrc0)
constexpr Isa GFX10{0xbe801f00u, 0xbe8003ffu, 0xbf820000u, 0xbe800400u}, GFX11{0xbe804700u, 0xbe8000ffu, 0xbfa00000u, 0xbe800100u};
constexpr uint32_t GETPC_MASK = 0xff80ffffu;   // s_getpc_b64 s[n:n+1]: n in bits 16..22
constexpr uint32_t SOP2_OP_MASK = 0xff800000u, S_ADD_U32 = 0x80000000u, S_ADDC_U32 = 0x82000000u;   // same in gfx10-gfx12
constexpr uint32_t S_NOP = 0xbf800000u, S_CODE_END = 0xbf9f0000u;
const Isa& isa(int family) { return family == 10 ? GFX10 : GFX11; }
// waits the compiler may put inside a getpc sequence: s_nop, gfx11 s_waitcnt_depctr / gfx12 s_wait_alu, gfx10 s_waitcnt_depctr
bool hazardWait(uint32_t w, int family) {
    const uint32_t hi = w & 0xffff0000u;
    return hi == S_NOP || (family >= 11 ? hi == 0xbf880000u : hi == 0xbfa30000u);
}

// s_add_u32/s_addc_u32 sN, sN, literal (either operand order)
bool sop2Literal(uint32_t w, uint32_t op, uint32_t reg) {
    if ((w & SOP2_OP_MASK) != op || ((w >> 16) & 0x7f) != reg) return false;
    const uint32_t s0 = w & 0xff, s1 = (w >> 8) & 0xff;
    return (s0 == reg && s1 == 0xff) || (s0 == 0xff && s1 == reg);
}

} // namespace

uint32_t gfx11Mach(uint32_t gfxTargetVersion) { return machOf(gfxTargetVersion); }

uint32_t machOf(uint32_t gfxTargetVersion) {
    switch (gfxTargetVersion) {
    case 100300: return 0x36;  // gfx1030 (RDNA2)
    case 100301: return 0x37;  // gfx1031
    case 100302: return 0x38;  // gfx1032
    case 100303: return 0x39;  // gfx1033
    case 100304: return 0x3e;  // gfx1034
    case 100305: return 0x3d;  // gfx1035
    case 100306: return 0x45;  // gfx1036
    case 110000: return 0x41;  // gfx1100 (RDNA3)
    case 110001: return 0x46;  // gfx1101
    case 110002: return 0x47;  // gfx1102
    case 110003: return 0x44;  // gfx1103
    // RDNA 3.5: the same gfx11 ISA and kernel descriptor, code built for its own target
    case 110500: return 0x43;  // gfx1150
    case 110501: return 0x4a;  // gfx1151
    case 110502: return 0x55;  // gfx1152
    case 110503: return 0x58;  // gfx1153
    case 120000: return 0x48;  // gfx1200 (RDNA4)
    case 120001: return 0x4e;  // gfx1201
    default: return 0;
    }
}

bool gfx115Mach(uint32_t mach) { return mach == 0x43 || mach == 0x4a || mach == 0x55 || mach == 0x58; }

int machFamily(uint32_t mach) {
    switch (mach) {
    case 0x36: case 0x37: case 0x38: case 0x39: case 0x3e: case 0x3d: case 0x45: return 10;
    case 0x41: case 0x46: case 0x47: case 0x44: case 0x43: case 0x4a: case 0x55: case 0x58: return 11;
    case 0x48: case 0x4e: return 12;
    default: return 0;
    }
}

bool Module::parse(const std::vector<uint8_t>& f) {
    mach = f.size() >= 64 ? rd<uint32_t>(f, 48) & 255 : 0;
    family = machFamily(mach);
    if (f.size() < 64 || std::memcmp(f.data(), "\x7f" "ELF", 4) || f[4] != 2 || f[5] != 1 || rd<uint16_t>(f, 18) != 224 || !family) {
        error = "requires a little-endian RDNA2/3/4 (gfx103x, gfx110x, gfx115x, gfx120x) AMDGPU ELF64 code object"; return false;
    }
    const Isa& I = isa(family);
    const uint64_t shoff = rd<uint64_t>(f, 0x28);
    const uint16_t shentsize = rd<uint16_t>(f, 0x3a), shnum = rd<uint16_t>(f, 0x3c), shstrndx = rd<uint16_t>(f, 0x3e);
    if (shentsize != 64 || shoff > f.size() || uint64_t(shnum) * 64 > f.size() - shoff || shstrndx >= shnum) { error = "bad section table"; return false; }
    std::vector<Section> s(shnum);
    for (uint32_t i = 0; i < shnum; ++i) {
        const size_t h = shoff + 64 * i;
        s[i].type = rd<uint32_t>(f, h + 4); s[i].flags = rd<uint64_t>(f, h + 8); s[i].addr = rd<uint64_t>(f, h + 16);
        s[i].offset = rd<uint64_t>(f, h + 24); s[i].size = rd<uint64_t>(f, h + 32); s[i].link = rd<uint32_t>(f, h + 40);
        if (s[i].link >= shnum || (s[i].type != SHT_NOBITS &&
            (s[i].offset > f.size() || s[i].size > f.size() - s[i].offset))) {
            error = "section outside code object"; return false;
        }
    }
    auto str = [&](uint32_t table, uint32_t at) {
        const uint64_t o = s[table].offset + at;
        std::string r;
        while (o + r.size() < f.size() && f[o + r.size()]) r += char(f[o + r.size()]);
        return r;
    };
    for (uint32_t i = 0; i < shnum; ++i) s[i].name = str(shstrndx, rd<uint32_t>(f, shoff + 64 * i));
    int text = -1, symtab = -1, rodata = -1;
    dataLo = ~0ull; dataHi = 0;
    for (uint32_t i = 0; i < shnum; ++i) {
        if (s[i].name == ".text") text = int(i);
        if (s[i].name == ".rodata" && s[i].type != SHT_NOBITS) rodata = int(i);
        if (s[i].type == SHT_SYMTAB || (s[i].type == SHT_DYNSYM && symtab < 0)) symtab = int(i);
        const bool data = s[i].name.rfind(".data", 0) == 0 || s[i].name.rfind(".bss", 0) == 0;
        if (data && (s[i].flags & SHF_ALLOC) && (s[i].flags & SHF_WRITE) && s[i].size) {
            dataLo = std::min(dataLo, s[i].addr); dataHi = std::max(dataHi, s[i].addr + s[i].size);
        }
    }
    if (text < 0 || symtab < 0) { error = "no .text or symbol table"; return false; }
    if (s[text].offset + s[text].size > f.size() || s[text].size % 4) { error = "bad .text"; return false; }
    textVa = s[text].addr;
    this->text.assign(f.begin() + s[text].offset, f.begin() + s[text].offset + s[text].size);
    if (dataHi > dataLo) {
        if (dataHi - dataLo > 256 * 1024 * 1024) { error = "code object data allocation exceeds limit"; return false; }
        dataInit.assign(dataHi - dataLo, 0);
        for (const auto& x : s) {
            if (x.type == SHT_NOBITS || !x.size || x.addr < dataLo || x.addr + x.size > dataHi) continue;
            if (!(x.name.rfind(".data", 0) == 0)) continue;
            std::memcpy(dataInit.data() + (x.addr - dataLo), f.data() + x.offset, x.size);
        }
    } else {
        dataLo = dataHi = 0;
    }
    // symbols: kernel descriptors (<name>.kd) and data objects
    const auto& st = s[symtab];
    for (uint64_t at = st.offset; at + 24 <= st.offset + st.size; at += 24) {
        const std::string name = str(st.link, rd<uint32_t>(f, at));
        const uint8_t info = f[at + 4];
        const uint16_t shndx = rd<uint16_t>(f, at + 6);
        const uint64_t value = rd<uint64_t>(f, at + 8), size = rd<uint64_t>(f, at + 16);
        if (name.empty() || shndx == 0 || shndx >= shnum) continue;
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0 && size == 64) {
            const uint64_t o = s[shndx].offset + (value - s[shndx].addr);
            if (o + 64 > f.size()) continue;
            Kernel k;
            k.name = name.substr(0, name.size() - 3);
            k.lds = rd<uint32_t>(f, o); k.scratch = rd<uint32_t>(f, o + 4); k.kernarg = rd<uint32_t>(f, o + 8);
            k.entry = value + uint64_t(rd<int64_t>(f, o + 16));
            k.rsrc3 = rd<uint32_t>(f, o + 44); k.rsrc1 = rd<uint32_t>(f, o + 48); k.rsrc2 = rd<uint32_t>(f, o + 52);
            k.props = rd<uint16_t>(f, o + 56); k.preload = rd<uint16_t>(f, o + 58);
            kernels[k.name] = k;
        } else if ((info & 0xf) == 1 /* STT_OBJECT */) {
            globals[name] = Global{value, size};
        }
    }
    // PC-relative global addressing
    const uint32_t* w = reinterpret_cast<const uint32_t*>(this->text.data());
    const size_t n = this->text.size() / 4;
    bool readsRodata = false;
    for (size_t i = 0; i + 4 < n; ++i) {
        if ((w[i] & GETPC_MASK) != I.getpc) continue;
        const uint32_t reg = (w[i] >> 16) & 0x7f;
        // s_getpc_b64; hazard waits (any number, anywhere in the sequence); gfx12: s_sext_i32_i16 s[n+1] (its PC is 48 bits);
        // s_add_u32 sN, sN, lit; s_addc_u32 sN+1, sN+1, lit
        size_t a = i + 1;
        auto skipWaits = [&] { while (a < n && hazardWait(w[a], family)) ++a; };
        skipWaits();
        if (family >= 12 && a < n && w[a] == (0xbe800f00u | ((reg + 1) << 16) | (reg + 1))) { ++a; skipWaits(); }
        if (a + 1 >= n || !sop2Literal(w[a], S_ADD_U32, reg)) { error = "unhandled s_getpc_b64 at text+" + std::to_string(4 * i); return false; }
        const uint32_t lo = w[a + 1];
        a += 2;
        skipWaits();
        if (a + 1 >= n || !sop2Literal(w[a], S_ADDC_U32, reg + 1)) { error = "unhandled s_getpc_b64 at text+" + std::to_string(4 * i); return false; }
        const uint32_t hi = w[a + 1];
        const size_t end = a + 2;   // the sequence is [i, end)
        const int64_t offset = int64_t((uint64_t(hi) << 32) | lo);
        const uint64_t target = textVa + 4 * (i + 1) + uint64_t(offset);
        if (target >= dataLo && target < dataHi) {
            sites.push_back(Site{uint32_t(i), reg, target, uint32_t(end - i)});
        } else if (rodata >= 0 && target >= s[rodata].addr && target < s[rodata].addr + s[rodata].size) {
            sites.push_back(Site{uint32_t(i), reg, target, uint32_t(end - i)});   // constant tables: mapped below, only for modules that read them
            readsRodata = true;
        } else if (target >= textVa && target < textVa + this->text.size()) {
            ++textGetpc;
        } else {
            error = "s_getpc_b64 at text+" + std::to_string(4 * i) + " addresses neither code nor data";
            return false;
        }
        i = end - 1;
    }
    if (readsRodata) {
        // m4r: the data image spans .rodata and the writable data, in whichever order the linker placed them
        const uint64_t rlo = s[rodata].addr, rhi = rlo + s[rodata].size;
        const bool hasData = dataHi > dataLo;
        const uint64_t lo = hasData ? std::min(rlo, dataLo) : rlo, hi = hasData ? std::max(rhi, dataHi) : rhi;
        if (hi - lo > 256 * 1024 * 1024) { error = "code object data allocation exceeds limit"; return false; }
        std::vector<uint8_t> image(hi - lo, 0);
        std::memcpy(image.data() + (rlo - lo), f.data() + s[rodata].offset, s[rodata].size);
        if (hasData) std::memcpy(image.data() + (dataLo - lo), dataInit.data(), dataInit.size());
        dataInit = std::move(image);
        dataLo = lo;
        dataHi = hi;
    }
    for (const auto& [name, k] : kernels) {
        if (k.entry < textVa || k.entry >= textVa + this->text.size() || (k.entry - textVa) % 4) { error = name + ": entry outside .text"; return false; }
        // Only the kernarg pointer is enabled and nothing is preloaded, except on RDNA2: there LLVM always enables the private
        // segment buffer (s[0:3], no architected flat scratch), so the kernarg pointer is s[4:5]; a kernel that uses no scratch
        // never reads s[0:3], and createPipeline copies the push constant to s[4:5] first. Any further user SGPRs are LLVM's
        // UserSGPRInit16Bug padding (gfx1100/gfx1102), unread, and the kernel's rsrc2 places its workgroup ids after them.
        const uint32_t user = (k.rsrc2 >> 1) & 0x1f;
        const uint32_t ids = uint32_t(std::popcount((k.rsrc2 >> 7) & 0xfu));  // tgid x/y/z, tg size (not the wave offset)
        const bool psb = k.props & 1u;
        if ((k.props & 0x7e) != (1u << 3) || (k.preload & 0x7f) || user < 2 || (user > 2 && !(psb && user == 6) && user + ids != 16)) {
            error = name + ": not a kernarg-only kernel"; return false;
        }
        if (psb && (k.scratch || (k.rsrc2 & 1u))) { error = name + ": uses scratch, which RDNA2 kernels cannot have here"; return false; }
        kernels[name].kernargSgpr = psb ? 4 : 0;
        // wave32 or wave64 (props bit 10); the donor pipeline is created with the same subgroup size
    }
    return true;
}

std::vector<uint32_t> Module::patchedText(uint64_t dataAddress) const {
    std::vector<uint32_t> w(text.size() / 4);
    std::memcpy(w.data(), text.data(), text.size());
    const Isa& I = isa(family);
    for (const auto& site : sites) {
        // the whole sequence becomes two literal moves of the data address, padded with s_nop (same length)
        const uint64_t a = dataAddress + (site.target - dataLo);
        const uint32_t at = site.word;
        w[at + 0] = I.movLit | (site.sdst << 16);
        w[at + 1] = uint32_t(a);
        w[at + 2] = I.movLit | ((site.sdst + 1) << 16);
        w[at + 3] = uint32_t(a >> 32);
        for (uint32_t k = 4; k < site.words; ++k) w[at + k] = S_NOP;
    }
    return w;
}

bool Device::load(VkDevice d, PFN_vkGetDeviceProcAddr proc) {
    device = d;
    createCache = reinterpret_cast<PFN_vkCreatePipelineCache>(proc(d, "vkCreatePipelineCache"));
    destroyCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(proc(d, "vkDestroyPipelineCache"));
    cacheData = reinterpret_cast<PFN_vkGetPipelineCacheData>(proc(d, "vkGetPipelineCacheData"));
    createCompute = reinterpret_cast<PFN_vkCreateComputePipelines>(proc(d, "vkCreateComputePipelines"));
    destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(proc(d, "vkDestroyPipeline"));
    createModule = reinterpret_cast<PFN_vkCreateShaderModule>(proc(d, "vkCreateShaderModule"));
    destroyModule = reinterpret_cast<PFN_vkDestroyShaderModule>(proc(d, "vkDestroyShaderModule"));
    return createCache && destroyCache && cacheData && createCompute && destroyPipeline && createModule && destroyModule;
}

namespace {

// radv_shader_binary: u32 type | ac_shader_config (56 bytes) | radv_shader_info | u32 total_size, then
// radv_shader_binary_legacy's code/exec/ir/disasm/stats/debug-info sizes and the data (stats, code, ir, disasm,
// debug info). Returns the offset of the legacy data (the end of the fixed header) or 0.
size_t legacyHeader(const std::vector<uint8_t>& b, size_t& totalAt, size_t& sizes) {
    if (b.size() < 128 || rd<uint32_t>(b, 0) != 0 /* RADV_BINARY_TYPE_LEGACY */) return 0;
    for (size_t p = 64; p + 32 <= b.size(); p += 4) {
        const uint32_t total = rd<uint32_t>(b, p);
        if (total != b.size()) continue;
        // radv_shader_binary is 8-byte aligned: the legacy sizes follow total_size directly or after 4 bytes
        for (size_t pad : {size_t(0), size_t(4)}) {
            const size_t q = p + 4 + pad;
            if (q + 24 > b.size()) continue;
            const uint32_t code = rd<uint32_t>(b, q), exec = rd<uint32_t>(b, q + 4), ir = rd<uint32_t>(b, q + 8),
                           disasm = rd<uint32_t>(b, q + 12), stats = rd<uint32_t>(b, q + 16), dbg = rd<uint32_t>(b, q + 20);
            if (!code || code % 4 || exec > code) continue;
            if (uint64_t(q) + 24 + code + ir + disasm + stats + dbg != total) continue;
            totalAt = p; sizes = q;
            return q + 24;
        }
    }
    return 0;
}

constexpr size_t CFG = 4;  // ac_shader_config offset
enum CfgField { NUM_SGPRS = 0, NUM_VGPRS = 4, SHARED_VGPRS = 8, SPILLED_SGPRS = 12, SPILLED_VGPRS = 16, LDS_SIZE = 20,
                FLOAT_MODE = 32, SCRATCH = 36, WGP_MODE = 40, RSRC1 = 44, RSRC2 = 48, RSRC3 = 52 };
constexpr uint32_t RSRC2_LDS_MASK = 0x1ffu << 15;

} // namespace

VkPipeline createPipeline(const Device& vk, VkPipelineLayout layout, const Module& module, const Kernel& kernel,
                          Launch launch, uint64_t dataAddress, uint32_t id, std::string& error) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(donor_spv); smi.pCode = donor_spv;
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vk.createModule(vk.device, &smi, nullptr, &shader) != VK_SUCCESS) { error = "donor shader module"; return VK_NULL_HANDLE; }
    const uint32_t spec[5] = {launch.bx, launch.by, launch.bz, std::max(1u, (kernel.lds + 3) / 4), id};
    VkSpecializationMapEntry entries[5];
    for (uint32_t i = 0; i < 5; ++i) entries[i] = {i, 4 * i, 4};
    VkSpecializationInfo si{5, entries, sizeof(spec), spec};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo wave{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    const bool wave64 = !(kernel.props & (1u << 10));
    wave.requiredSubgroupSize = wave64 ? 64 : 32;
    const uint32_t vgran = wave64 ? 4 : 8;   // RSRC1 VGPR field unit (gfx10+: 8 for wave32, 4 for wave64)
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &wave, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", &si};
    ci.layout = layout;
    VkPipeline result = VK_NULL_HANDLE;
    VkPipelineCache donorCache = VK_NULL_HANDLE, patchedCache = VK_NULL_HANDLE;
    VkPipeline donor = VK_NULL_HANDLE;
    std::vector<uint8_t> data, out;
    size_t size = 0;
    auto fail = [&](const std::string& why) { error = kernel.name + ": " + why; };
    do {
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (vk.createCache(vk.device, &cci, nullptr, &donorCache) != VK_SUCCESS) { fail("donor cache"); break; }
        if (vk.createCompute(vk.device, donorCache, 1, &ci, nullptr, &donor) != VK_SUCCESS) { fail("donor pipeline"); break; }
        if (vk.cacheData(vk.device, donorCache, &size, nullptr) != VK_SUCCESS || !size) { fail("cache size"); break; }
        data.resize(size);
        if (vk.cacheData(vk.device, donorCache, &size, data.data()) != VK_SUCCESS) { fail("cache data"); break; }
        data.resize(size);
        // vk_pipeline_cache: header (header_size first), u32 count, then {u32 type, u32 key size, u32 data size,
        // key, align 8, data} per object
        const uint32_t headerSize = rd<uint32_t>(data, 0);
        if (headerSize < 16 || headerSize + 4 > data.size()) { fail("cache header"); break; }
        const uint32_t count = rd<uint32_t>(data, headerSize);
        out.assign(data.begin(), data.begin() + headerSize + 4);
        size_t at = headerSize + 4;
        int shaders = 0;
        bool bad = false;
        for (uint32_t e = 0; e < count && !bad; ++e) {
            if (at + 12 > data.size()) { bad = true; break; }
            const uint32_t type = rd<uint32_t>(data, at), keySize = rd<uint32_t>(data, at + 4), dataSize = rd<uint32_t>(data, at + 8);
            size_t keyAt = at + 12, dataAt = (keyAt + keySize + 7) & ~size_t(7);
            if (dataAt + dataSize > data.size()) { bad = true; break; }
            std::vector<uint8_t> object(data.begin() + dataAt, data.begin() + dataAt + dataSize);
            size_t q = 0, totalAt = 0;
            const size_t h = legacyHeader(object, totalAt, q);
            if (h) {
                // the donor's config: LDS and user SGPRs must be what the kernel expects
                const uint32_t donorRsrc2 = rd<uint32_t>(object, CFG + RSRC2);
                // the donor's user SGPRs: the push constant inlined, at s[0:1] (RDNA3) or after a ring-offsets pair
                if (((donorRsrc2 >> 1) & 0x1f) != 2 && ((donorRsrc2 >> 1) & 0x1f) != 4) {
                    fail("donor user SGPRs not 2 or 4 (push constant not inlined)"); bad = true; break;
                }
                // Fingerprint of the layout this code was written against (Mesa 26.2): a config whose fields do not
                // agree with each other and with what the donor declares means the struct moved; refuse rather than
                // patch the wrong words.
                {
                    const uint32_t sgprs = rd<uint32_t>(object, CFG + NUM_SGPRS), vgprs = rd<uint32_t>(object, CFG + NUM_VGPRS);
                    const uint32_t lds = rd<uint32_t>(object, CFG + LDS_SIZE), scratch = rd<uint32_t>(object, CFG + SCRATCH);
                    const uint32_t rsrc1 = rd<uint32_t>(object, CFG + RSRC1), fmode = rd<uint32_t>(object, CFG + FLOAT_MODE);
                    const uint32_t declared = std::max(1u, (kernel.lds + 3) / 4) * 4;
                    const bool ldsOk = (lds >= declared && lds < declared + 1024) || (kernel.lds == 0 && lds == 0);
                    const bool sane = sgprs && sgprs <= 128 && vgprs && vgprs <= 256 && ((rsrc1 & 0x3f) + 1) * vgran >= vgprs &&
                                      ((rsrc1 & 0x3f) + 1) * vgran < vgprs + vgran && ldsOk &&
                                      ((donorRsrc2 & RSRC2_LDS_MASK) != 0) == (lds != 0) && scratch == 0 && fmode < 256 &&
                                      object[CFG + WGP_MODE] <= 1 && ((rsrc1 >> 12) & 0xff) == fmode;
                    if (!sane) {
                        char fields[160];
                        std::snprintf(fields, sizeof(fields), " [sgprs %u vgprs %u rsrc1 %#x rsrc2 %#x lds %u/%u scratch %u fmode %u wgp %u]",
                                      sgprs, vgprs, rsrc1, donorRsrc2, lds, declared, scratch, fmode, object[CFG + WGP_MODE]);
                        fail(std::string("unrecognized RADV shader binary layout (Mesa changed?)") + fields); bad = true; break;
                    }
                }
                std::vector<uint8_t> patched(object.begin(), object.begin() + h);
                // code: s_branch to the entry, padding to 256 bytes, .text (its 256-byte alignment kept), tail
                std::vector<uint32_t> code(64, S_NOP);
                const uint64_t entryWord = 64 + (kernel.entry - module.textVa) / 4;
                // the push constant (the kernarg pointer) is the donor's last user SGPR pair; the kernel wants it at
                // s[kernargSgpr]: one s_mov_b64 first when they differ, then the branch to the entry
                const uint32_t donorUser = (donorRsrc2 >> 1) & 0x1f, kernelUser = (kernel.rsrc2 >> 1) & 0x1f;
                const uint32_t pushAt = donorUser - 2;
                if (pushAt + 2 > kernelUser) { fail("the donor's push constant lies beyond the kernel's user SGPRs"); bad = true; break; }
                uint32_t at = 0;
                if (pushAt != kernel.kernargSgpr) code[at++] = isa(module.family).movB64 | (kernel.kernargSgpr << 16) | pushAt;
                const int64_t simm = int64_t(entryWord) - int64_t(at) - 1;
                if (simm > 32767) { fail("entry beyond s_branch range"); bad = true; break; }
                code[at] = isa(module.family).branch | uint32_t(simm & 0xffff);
                const auto text = module.patchedText(dataAddress);
                code.insert(code.end(), text.begin(), text.end());
                code.insert(code.end(), 64, S_CODE_END);
                const uint32_t codeBytes = uint32_t(code.size() * 4);
                wr<uint32_t>(patched, CFG + NUM_SGPRS, 106);
                wr<uint32_t>(patched, CFG + NUM_VGPRS, ((kernel.rsrc1 & 0x3f) + 1) * vgran);
                wr<uint32_t>(patched, CFG + SHARED_VGPRS, 0);
                wr<uint32_t>(patched, CFG + SPILLED_SGPRS, 0);
                wr<uint32_t>(patched, CFG + SPILLED_VGPRS, 0);
                wr<uint32_t>(patched, CFG + FLOAT_MODE, (kernel.rsrc1 >> 12) & 0xff);
                wr<uint32_t>(patched, CFG + SCRATCH, (kernel.scratch * (wave64 ? 64 : 32) + 255) / 256 * 256);
                patched[CFG + WGP_MODE] = uint8_t((kernel.rsrc1 >> 29) & 1);
                wr<uint32_t>(patched, CFG + RSRC1, kernel.rsrc1);
                wr<uint32_t>(patched, CFG + RSRC2, (kernel.rsrc2 & ~RSRC2_LDS_MASK) | (donorRsrc2 & RSRC2_LDS_MASK));
                wr<uint32_t>(patched, CFG + RSRC3, kernel.rsrc3);
                // total_size precedes the legacy sizes (code, exec, ir, disasm, stats, debug info)
                wr<uint32_t>(patched, totalAt, uint32_t(h + codeBytes));
                wr<uint32_t>(patched, q, codeBytes);
                wr<uint32_t>(patched, q + 4, codeBytes);
                for (int k = 2; k <= 5; ++k) wr<uint32_t>(patched, q + 4 * k, 0);
                patched.resize(h + codeBytes);
                std::memcpy(patched.data() + h, code.data(), codeBytes);
                object.swap(patched);
                ++shaders;
            }
            const size_t base = out.size();
            out.resize(base + 12);
            wr<uint32_t>(out, base, type); wr<uint32_t>(out, base + 4, keySize); wr<uint32_t>(out, base + 8, uint32_t(object.size()));
            out.insert(out.end(), data.begin() + keyAt, data.begin() + keyAt + keySize);
            while (out.size() % 8) out.push_back(0);
            out.insert(out.end(), object.begin(), object.end());
            at = dataAt + dataSize;
        }
        if (bad) { if (error.empty()) fail("cache data layout"); break; }
        if (shaders != 1) { fail("expected one shader object in the donor cache, found " + std::to_string(shaders)); break; }
        cci.initialDataSize = out.size(); cci.pInitialData = out.data();
        if (vk.createCache(vk.device, &cci, nullptr, &patchedCache) != VK_SUCCESS) { fail("patched cache"); break; }
        if (vk.createCompute(vk.device, patchedCache, 1, &ci, nullptr, &result) != VK_SUCCESS) { result = VK_NULL_HANDLE; fail("pipeline"); break; }
    } while (false);
    if (donor) vk.destroyPipeline(vk.device, donor, nullptr);
    if (donorCache) vk.destroyCache(vk.device, donorCache, nullptr);
    if (patchedCache) vk.destroyCache(vk.device, patchedCache, nullptr);
    vk.destroyModule(vk.device, shader, nullptr);
    return result;
}

} // namespace vkhip
