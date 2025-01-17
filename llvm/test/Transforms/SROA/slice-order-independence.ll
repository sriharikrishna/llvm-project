; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

; Check that the chosen type for a split is independent from the order of
; slices even in case of types that are skipped because their width is not a
; byte width multiple
define void @skipped_inttype_first(ptr) {
; CHECK-LABEL: @skipped_inttype_first
; CHECK: alloca ptr
  %arg = alloca { ptr, i32 }, align 8
  %2 = bitcast ptr %0 to ptr
  %3 = bitcast ptr %arg to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %3, ptr align 8 %2, i32 16, i1 false)
  %b = getelementptr inbounds { ptr, i32 }, ptr %arg, i64 0, i32 0
  %b0 = load i63, ptr %b
  %b1 = load ptr, ptr %b
  ret void
}

define void @skipped_inttype_last(ptr) {
; CHECK-LABEL: @skipped_inttype_last
; CHECK: alloca ptr
  %arg = alloca { ptr, i32 }, align 8
  %2 = bitcast ptr %0 to ptr
  %3 = bitcast ptr %arg to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %3, ptr align 8 %2, i32 16, i1 false)
  %b = getelementptr inbounds { ptr, i32 }, ptr %arg, i64 0, i32 0
  %b1 = load ptr, ptr %b
  %b0 = load i63, ptr %b
  ret void
}
