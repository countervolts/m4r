// m4r: hipvk's vkCmdBlitImage between formats (same size, nearest): the hardware's format conversion, image load then store.
// Arguments: a 64-byte record (source image descriptor, destination image descriptor), width, height.
// SPDX-License-Identifier: GPL-3.0-only
typedef float float4v __attribute__((ext_vector_type(4)));
typedef __amdgpu_texture_t tex_t;

__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void hipvk_blit(constant const char* rec, uint w, uint h) {
    const int x = __builtin_amdgcn_workgroup_id_x() * 8 + (int)__builtin_amdgcn_workitem_id_x();
    const int y = __builtin_amdgcn_workgroup_id_y() * 8 + (int)__builtin_amdgcn_workitem_id_y();
    if (x >= (int)w || y >= (int)h) return;
    const float4v v = __builtin_amdgcn_image_load_2d_v4f32_i32(15, x, y, *(constant const tex_t*)rec, 0, 0);
    __builtin_amdgcn_image_store_2d_v4f32_i32(v, 15, x, y, *(constant const tex_t*)(rec + 32), 0, 0);
}
