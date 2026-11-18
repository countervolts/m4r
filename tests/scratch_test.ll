; A small AIR-style kernel (written for m4r, not from MetalFX) that keeps 7200 bytes per lane in private memory:
;   device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]
;   out[gid] = sum over i < 1800 of (gid * 7 + i), computed through a private array (volatile, so it stays in memory)
; tests/test_scratch.cpp dispatches it with many waves to check that scratch memory works in injected pipelines.
; SPDX-License-Identifier: GPL-3.0-only
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.2.0"

@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

define void @scratch_test(i32 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
entry:
  %arr = alloca [1800 x i32], align 4
  %mul = mul i32 %1, 7
  br label %fill

fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %v = add i32 %mul, %i
  %p = getelementptr inbounds [1800 x i32], [1800 x i32]* %arr, i32 0, i32 %i
  store volatile i32 %v, i32* %p, align 4
  %i1 = add nuw nsw i32 %i, 1
  %c = icmp eq i32 %i1, 1800
  br i1 %c, label %sum, label %fill

sum:
  %j = phi i32 [ 0, %fill ], [ %j1, %sum ]
  %acc = phi i32 [ 0, %fill ], [ %acc1, %sum ]
  %rev = sub i32 1799, %j
  %q = getelementptr inbounds [1800 x i32], [1800 x i32]* %arr, i32 0, i32 %rev
  %x = load volatile i32, i32* %q, align 4
  %acc1 = add i32 %acc, %x
  %j1 = add nuw nsw i32 %j, 1
  %d = icmp eq i32 %j1, 1800
  br i1 %d, label %done, label %sum

done:
  %zp = zext i32 %1 to i64
  %o = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %zp
  store i32 %acc1, i32 addrspace(1)* %o, align 4
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.version = !{!10}
!air.language_version = !{!11}
!air.kernel = !{!15}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!10 = !{i32 2, i32 9, i32 0}
!11 = !{!"Metal", i32 3, i32 1, i32 0}
!15 = !{void (i32 addrspace(1)*, i32)* @scratch_test, !16, !17}
!16 = !{}
!17 = !{!18, !19}
!18 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!19 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
