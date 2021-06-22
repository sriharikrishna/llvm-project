#ifndef _INTEROP_H_
#define _INTEROP_H_

#include <assert.h>
#include "omptarget.h"
/*
#include "../src/device.h"
#include "omptargetplugin.h"
#include "../src/private.h"
#include "../src/rtl.h"
*/
typedef struct kmp_tasking_flags { /* Total struct must be exactly 32 bits */
  /* Compiler flags */ /* Total compiler flags must be 16 bits */
  unsigned tiedness : 1; /* task is either tied (1) or untied (0) */
  unsigned final : 1; /* task is final(1) so execute immediately */
  unsigned merged_if0 : 1; // no __kmpc_task_{begin/complete}_if0 calls in if0
  unsigned destructors_thunk : 1; // set if the compiler creates a thunk to
  unsigned proxy : 1; // task is a proxy task (it will be executed outside the
  unsigned priority_specified : 1; // set if the compiler provides priority
  unsigned detachable : 1; // 1 == can detach */
  unsigned unshackled : 1; /* 1 == unshackled task */
  unsigned target : 1; /* 1 == target task */
  unsigned reserved : 7; /* reserved for compiler use */
  unsigned tasktype : 1; /* task is either explicit(1) or implicit (0) */
  unsigned task_serial : 1; // task is executed immediately (1) or deferred (0)
  unsigned tasking_ser : 1; // all tasks in team are either executed immediately
  unsigned team_serial : 1; // entire team is serial (1) [1 thread] or parallel
  unsigned started : 1; /* 1==started, 0==not started     */
  unsigned executing : 1; /* 1==executing, 0==not executing */
  unsigned complete : 1; /* 1==complete, 0==not complete   */
  unsigned freed : 1; /* 1==freed, 0==allocated        */
  unsigned native : 1; /* 1==gcc-compiled task, 0==intel */
  unsigned reserved31 : 7; /* reserved for library use */
} kmp_tasking_flags_t;


typedef enum omp_interop_backend_type_t {
  // reserve 0
  omp_interop_backend_type_cuda_1 = 1,
} omp_interop_backend_type_t;

typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_platform,
  kmp_interop_type_device,
  kmp_interop_type_tasksync,
} kmp_interop_type_t;

extern "C" {

/// The interop value type, aka. the interop object.
typedef struct omp_interop_val_t {
  /// Device and interop-type are determined at construction time and fix.
  omp_interop_val_t(intptr_t device_id, kmp_interop_type_t interop_type)
      : interop_type(interop_type), device_id(device_id) {
    assert(interop_type != kmp_interop_type_unknown);
    assert(!async_info);
    assert(!device_info.Device);
    assert(!device_info.Context);
  }
  const char *err_str = nullptr;
  __tgt_async_info *async_info = nullptr;
  __tgt_device_info device_info;
  const kmp_interop_type_t interop_type;
  const intptr_t device_id;
  const intptr_t vendor_id = 1; // LLVM?
  const intptr_t backend_type_id = omp_interop_backend_type_cuda_1;
} omp_interop_val_t;

}

/*
#ifdef __cplusplus
 extern "C" {
#endif
void __kmpc_interop_init(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t **interop_ptr,
                         kmp_interop_type_t interop_type, kmp_int32 device_id,
                         kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                         kmp_int32 ndeps_noalias,
                         kmp_depend_info_t *noalias_dep_list);

void __kmpc_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t **interop_ptr,
                         kmp_interop_type_t interop_type, kmp_int32 device_id,
                         kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                         kmp_int32 ndeps_noalias,
                         kmp_depend_info_t *noalias_dep_list);
void __kmpc_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t **interop_ptr,
                            kmp_interop_type_t interop_type,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list,
                            kmp_int32 ndeps_noalias,
                            kmp_depend_info_t *noalias_dep_list);

#ifdef __cplusplus
 }
#endif
*/
#endif
