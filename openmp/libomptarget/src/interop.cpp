//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// TODO: The buildsystem seems to not include the `omp.h` of the current build
/// but the system `omp.h` :( Also, omp.h has obviously conflicting declarations
/// with other headers, because, why not.
//#include <omp.h>
//#include "../../../projects/openmp/runtime/src/omp.h"
#include "../../../build/runtimes/runtimes-bins/openmp/runtime/src/omp.h"
//#include "../include/omptarget.h"
#include "../include/interop.h"

#include "device.h"
#include "omptargetplugin.h"
#include "private.h"
#include "rtl.h"
//#include "../deviceRTLs/interface.h"
//#include "../../runtime/src/kmp.h"
#include <cassert>
/*
typedef struct kmp_tasking_flags { //  Total struct must be exactly 32 bits 
  // Compiler flags  Total compiler flags must be 16 bits 
  unsigned tiedness : 1; // task is either tied (1) or untied (0) 
  unsigned final : 1; // task is final(1) so execute immediately 
  unsigned merged_if0 : 1; // no __kmpc_task_{begin/complete}_if0 calls in if0
  unsigned destructors_thunk : 1; // set if the compiler creates a thunk to
  unsigned proxy : 1; // task is a proxy task (it will be executed outside the
  unsigned priority_specified : 1; // set if the compiler provides priority
  unsigned detachable : 1; // 1 == can detach 
  unsigned unshackled : 1; // 1 == unshackled task 
  unsigned target : 1; // 1 == target task 
  unsigned reserved : 7; // reserved for compiler use 
  unsigned tasktype : 1; // task is either explicit(1) or implicit (0) 
  unsigned task_serial : 1; // task is executed immediately (1) or deferred (0)
  unsigned tasking_ser : 1; // all tasks in team are either executed immediately
  unsigned team_serial : 1; // entire team is serial (1) [1 thread] or parallel
  unsigned started : 1; // 1==started, 0==not started     
  unsigned executing : 1; // 1==executing, 0==not executing 
  unsigned complete : 1; // 1==complete, 0==not complete   
  unsigned freed : 1; // 1==freed, 0==allocated        
  unsigned native : 1; // 1==gcc-compiled task, 0==intel 
  unsigned reserved31 : 7; // reserved for library use 
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
*/
static omp_interop_rc_t
__kmpc_interop_get_property_err_type(omp_interop_property_t property) {
  switch (property) {
  case omp_ipr_fr_id:
    return omp_irc_type_int;
  case omp_ipr_fr_name:
    return omp_irc_type_str;
  case omp_ipr_vendor:
    return omp_irc_type_int;
  case omp_ipr_vendor_name:
    return omp_irc_type_str;
  case omp_ipr_device_num:
    return omp_irc_type_int;
  case omp_ipr_platform:
    return omp_irc_type_int;
  case omp_ipr_device:
    return omp_irc_type_ptr;
  case omp_ipr_device_context:
    return omp_irc_type_ptr;
  case omp_ipr_targetsync:
    return omp_irc_type_ptr;
//  case omp_ipr_first:
//    return omp_irc_type_ptr;
  };
  return omp_irc_no_value;
}

static void __kmpc_interop_type_mismatch(omp_interop_property_t property,
                                         int *err) {
  if (err)
    *err = __kmpc_interop_get_property_err_type(property);
}

static const char *__kmpc_interop_vendor_id_to_str(intptr_t vendor_id) {
  return "CUDA_driver";
}

template <typename PropertyTy>
static PropertyTy __kmpc_interop_get_property(omp_interop_val_t &interop_val,
                                              omp_interop_property_t property,
                                              int *err);

template <>
intptr_t __kmpc_interop_get_property<intptr_t>(omp_interop_val_t &interop_val,
                                               omp_interop_property_t property,
                                               int *err) {
	      
  switch (property) {
  case omp_ipr_fr_id:
    //printf("II1");
    return interop_val.backend_type_id;
  case omp_ipr_vendor:
    //printf("II2");
    return interop_val.vendor_id;
  case omp_ipr_device_num:
    //printf("II3");
    return interop_val.device_id;
  default:
    //printf("II4");
    //printf("Integer property not handled in switch \n");
    ;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_int &&
           //"Integer property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return 0;
}

template <>
const char *__kmpc_interop_get_property<const char *>(
    omp_interop_val_t &interop_val, omp_interop_property_t property, int *err) {
  switch (property) {
  case omp_ipr_fr_id:
	  //printf("YY fr_id ");
	  //printf(" \n __kmpc_interop_get_property  case omp_ipr_fr_id \n ");
    return interop_val.interop_type == kmp_interop_type_tasksync ? "tasksync"
                                                                 : "device+context";
  case omp_ipr_vendor_name:
	  //printf("YY vendor_name");
	  //printf(" \n __kmpc_interop_get_property  case omp_ipr_vendor_name \n ");
    return __kmpc_interop_vendor_id_to_str(interop_val.vendor_id);
  default:
	  //printf("YY not handled");
    //printf("Const char *not handled in switch \n");
    __kmpc_interop_type_mismatch(property, err);
    return nullptr;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_str &&
           //"String property not handled in switch");
  }
}
/*
template <>
const char *__kmpc_interop_get_property<const char *>(
    omp_interop_val_t &interop_val, omp_interop_property_t property, int *err) {
  switch (property) {
  case omp_ipr_fr_id:
    return interop_val.interop_type == kmp_interop_type_tasksync ? "tasksync"
                                                                 : "device+context";
  case omp_ipr_vendor_name:
    return __kmpc_interop_vendor_id_to_str(interop_val.vendor_id);
  default:
    ;
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_str &&
           //"String property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return nullptr;
}
*/
template <>
void *__kmpc_interop_get_property<void *>(omp_interop_val_t &interop_val,
                                          omp_interop_property_t property,
                                          int *err) {
  //printf("In __kmpc_interop_get_property \n");
  switch (property) {
  case omp_ipr_device:
    //printf("XX device");
    if (interop_val.device_info.Device)
      return interop_val.device_info.Device;
    //*err = omp_interop_err_failure_str;
    *err = omp_irc_no_value;
    return const_cast<char *>(interop_val.err_str);
  case omp_ipr_device_context:
    //printf("XX device_context");
    return interop_val.device_info.Context;
  case omp_ipr_targetsync:
    //printf("XX targetsync");
    if(interop_val.async_info==NULL)
	    printf("interop_val.async_info==NULL \n");
    return interop_val.async_info->Queue;
  default:
    ;
    //printf("Pointer property not handled in switch %d \n", property);
    //assert(__kmpc_interop_get_property_err_type(property) !=
               //omp_interop_err_result_is_str &&
           //"Pointer property not handled in switch");
  }
  __kmpc_interop_type_mismatch(property, err);
  return nullptr;
}

static bool __kmpc_interop_get_property_check(omp_interop_val_t **interop_ptr,
                                              omp_interop_property_t property,
                                              int *err) {
  //printf("\n __kmpc_interop_get_property_check() Step1 \n");
  if (err)
    *err = omp_irc_success;
  //printf("\n __kmpc_interop_get_property_check() Step2 \n");
  if (!interop_ptr) {
    if (err)
      *err = omp_irc_empty;
    return false;
  }
  //printf("\n __kmpc_interop_get_property_check() Step3 \n");
  if (property >= 0 || property < omp_ipr_first) {
    if (err)
      *err = omp_irc_out_of_range;
    return false;
  }
  //printf("\n __kmpc_interop_get_property_check() Step4 \n");
  if (property == omp_ipr_targetsync &&
      (*interop_ptr)->interop_type != kmp_interop_type_tasksync) {
    if (err)
      *err = omp_irc_other;
    /*if (property == omp_ipr_targetsync)
      printf(" returning FALSE property == omp_ipr_targetsync \n");
    if ((*interop_ptr)->interop_type != kmp_interop_type_tasksync) 
      printf(" returning FALSE ((*interop_ptr)->interop_type != kmp_interop_type_tasksync) \n");*/
    return false;
  }
  //printf("\n __kmpc_interop_get_property_check() Step5 \n");
  if ((property == omp_ipr_device || property == omp_ipr_device_context) &&
      (*interop_ptr)->interop_type == kmp_interop_type_tasksync) {
    if (err)
      *err = omp_irc_other;
    return false;
  }
  //printf("\n __kmpc_interop_get_property_check() Step6 \n");
  return true;
}


#define __OMP_GET_INTEROP_TY(RETURN_TYPE, SUFFIX)                              \
RETURN_TYPE omp_get_interop_##SUFFIX(const omp_interop_t interop,              \
                                 omp_interop_property_t property_id,           \
                                 int *err) {                                   \
    omp_interop_val_t *interop_val = (omp_interop_val_t*) interop;             \
    assert((interop_val)->interop_type == kmp_interop_type_tasksync);\
    if (!__kmpc_interop_get_property_check(&interop_val, property_id, err)){    \
      return (RETURN_TYPE)(0);                                                 \
    }\
    return __kmpc_interop_get_property<RETURN_TYPE>(*interop_val, property_id, \
                                                    err);                      \
}
__OMP_GET_INTEROP_TY(intptr_t, int)
__OMP_GET_INTEROP_TY(void *, ptr)
__OMP_GET_INTEROP_TY(const char *, str)
#undef __OMP_GET_INTEROP_TY


#define __OMP_GET_INTEROP_TY3(RETURN_TYPE, SUFFIX)                             \
RETURN_TYPE omp_get_interop_##SUFFIX(const omp_interop_t interop,              \
                                 omp_interop_property_t property_id) {         \
    int err;                                                                  \
    omp_interop_val_t *interop_val = (omp_interop_val_t*) interop;             \
    if (!__kmpc_interop_get_property_check(&interop_val, property_id, &err)){    \
      return (RETURN_TYPE)(0); \
    }\
    return nullptr;\
    return __kmpc_interop_get_property<RETURN_TYPE>(*interop_val, property_id, \
                                                    &err);                      \
}
__OMP_GET_INTEROP_TY3(const char*, name)
__OMP_GET_INTEROP_TY3(const char*, type_desc)
__OMP_GET_INTEROP_TY3(const char*, rc_desc)
#undef __OMP_GET_INTEROP_TY3

typedef int64_t kmp_int64;

typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32, void *);

int waitForDeps(DeviceTy &Device, __tgt_async_info *AsyncInfo);
int recordEvent(DeviceTy &Device, __tgt_async_info *AsyncInfo);
int queryAndWait(DeviceTy &Device, __tgt_async_info *AsyncInfo);

extern "C" {
int __kmpc_set_async_info(kmp_int32 device_id, void *async_info);

//SHK: Copied from https://github.com/llvm/llvm-project/commit/ad95c48783a94b6ba126c2184205086c1ae8dd7c
int __kmpc_set_async_info(kmp_int32 device_id, void *async_info) {
   __tgt_async_info * AsyncInfo = (__tgt_async_info *) async_info;
   //initAsyncInfo(device_id, &AsyncInfo);
  /*int gtid = __kmp_get_gtid();
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_depnode_t *dep = thread->th.th_current_task->td_depnode;
  if (!dep)
    return 0;
  KMP_ATOMIC_ST_REL(&dep->dn.async_info,
                    reinterpret_cast<uintptr_t>(async_info));*/
  return 1;
}
/*
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                  kmp_int32 flags, size_t sizeof_kmp_task_t,
                                  size_t sizeof_shareds,
                                  kmp_routine_entry_t task_entry) {
  kmp_task_t *retval;
  kmp_tasking_flags_t *input_flags = (kmp_tasking_flags_t *)&flags;
  __kmp_assert_valid_gtid(gtid);
  input_flags->native = FALSE;
  // __kmp_task_alloc() sets up all other runtime flags
  KA_TRACE(10, ("__kmpc_omp_task_alloc(enter): T#%d loc=%p, flags=(%s %s %s) "
                "sizeof_task=%ld sizeof_shared=%ld entry=%p\n",
                gtid, loc_ref, input_flags->tiedness ? "tied  " : "untied",
                input_flags->proxy ? "proxy" : "",
                input_flags->detachable ? "detachable" : "", sizeof_kmp_task_t,
                sizeof_shareds, task_entry));

  retval = __kmp_task_alloc(loc_ref, gtid, input_flags, sizeof_kmp_task_t,
                            sizeof_shareds, task_entry);

  KA_TRACE(20, ("__kmpc_omp_task_alloc(exit): T#%d retval %p\n", gtid, retval));

  return retval;
}
*/
}

typedef union kmp_cmplrdata {
  kmp_int32 priority; /**< priority specified by user for the task */
  kmp_routine_entry_t
      destructors; /* pointer to function to invoke deconstructors of
                      firstprivate C++ objects */
  /* future data */
} kmp_cmplrdata_t;
typedef struct kmp_task { /* GEH: Shouldn't this be aligned somehow? */
  void *shareds; /**< pointer to block of pointers to shared vars   */
  kmp_routine_entry_t
      routine; /**< pointer to routine to call for executing task */
  kmp_int32 part_id; /**< part id for the task                          */
  kmp_cmplrdata_t
      data1; /* Two known optional additions: destructors and priority */
  kmp_cmplrdata_t data2; /* Process destructors first, priority second */
  /* future data */
  /*  private vars  */
} kmp_task_t;

kmp_int32 __fake_task_destroy(kmp_int32 i, void *t) {
    //printf("Fake DESTORY %i %p\n", i, t);
    kmp_task_t *task = (kmp_task_t*)t;
    DeviceTy &Device = *(((DeviceTy**)task->shareds)[0]);
    __tgt_async_info *AsyncInfo = ((__tgt_async_info**)task->shareds)[1];


    // Attach the async info to current task such that all dependent tasks can
    // start wait for the event if there is any dependency
    bool HasDependency = __kmpc_set_async_info(i, AsyncInfo);
    (void)HasDependency;
    //printf("HasDeps: %i\n", HasDependency);

    int Ret = 0;
    Ret = recordEvent(Device, AsyncInfo);
    //printf("Ret: %i\n", Ret);
    assert(Ret == OFFLOAD_SUCCESS);

    Ret = queryAndWait(Device, AsyncInfo);
    //printf("Ret: %i\n", Ret);
    assert(Ret == OFFLOAD_SUCCESS);
    return Ret;
}

kmp_int32 __fake_task_use(kmp_int32 i, void *t) {
    //printf("Fake USE %i %p\n", i, t);
    kmp_task_t *task = (kmp_task_t*)t;
    DeviceTy &Device = *(((DeviceTy**)task->shareds)[0]);
    __tgt_async_info *AsyncInfo = ((__tgt_async_info**)task->shareds)[1];

    int Ret = 0;
    Ret = waitForDeps(Device, AsyncInfo);
    //printf("Ret: %i %p\n", Ret, &Device);
    assert(Ret == OFFLOAD_SUCCESS);

    __fake_task_destroy(i, t);
    return Ret;
}

/*
extern "C" {
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                         kmp_int32 flags,
                                         size_t sizeof_kmp_task_t,
                                         size_t sizeof_shareds,
                                         kmp_routine_entry_t task_entry,
                                         kmp_int64 device_id);
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list);
}
*/
#ifdef __cplusplus
 extern "C" {
#endif
void __kmpc_interop_init(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t *&interop_ptr,
                         kmp_interop_type_t interop_type  ,kmp_int32 device_id,
                         kmp_int64 ndeps, kmp_depend_info_t *dep_list, kmp_int32 have_nowait) {
  //kmp_int32 ndeps = 0;
  //kmp_depend_info_t *dep_list = NULL;
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;
  printf("In __kmpc_interop_init \n");
  printf("__kmpc_interop_init(): interop_ptr %i \n", interop_ptr);
  //SHK assert(interop_ptr && "Cannot initialize nullptr!");
  //SHK assert(*interop_ptr == NULL && "*interop_ptr is not NULL");
  assert(interop_type != kmp_interop_type_unknown &&
         "Cannot initialize with unknown interop_type!");
  printf("__kmpc_interop_init(): device_id %d interop_type %d \n", device_id, interop_type);
  if (device_id == -1){
    printf("__kmpc_interop_init(): setting device_id to %d \n", omp_get_default_device());
    device_id = omp_get_default_device();
  }
  //SHK *interop_ptr = new omp_interop_val_t(device_id, interop_type);
  printf("__kmpc_interop_init(): %i \n", interop_ptr);	  
  interop_ptr = new omp_interop_val_t(device_id, interop_type);
  printf("__kmpc_interop_init(): %i \n", interop_ptr);	  

  if (device_id == omp_get_initial_device()) {
    printf("__kmpc_interop_init(): Unhandled case: device_id == omp_get_initial_device(); This implies that the device is the host\n");
    assert(device_id != omp_get_initial_device());
    return;
  }

  if (!device_is_ready(device_id)) {
    printf("__kmpc_interop_init(): Device not ready!\n");	  
    //SHK (*interop_ptr)->err_str = "Device not ready!";
    interop_ptr->err_str = "Device not ready!";
    return;
  }

  DeviceTy &Device = PM->Devices[device_id];
  if (interop_type == kmp_interop_type_tasksync) {
	  printf("__kmpc_interop_init(): (interop_type == kmp_interop_type_tasksync)\n");
//#ifdef SYNC_INTEROP
#if 1
	  printf("__kmpc_interop_init(): call __kmpc_omp_wait_deps()\n");
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
#endif
  }

  if (interop_type == kmp_interop_type_tasksync) {
    if (!Device.RTL || !Device.RTL->init_async_info ||
        //SHK Device.RTL->init_async_info(device_id, &(*interop_ptr)->async_info)) {
        Device.RTL->init_async_info(device_id, &(interop_ptr)->async_info)) {
      // error
      //SHK delete *interop_ptr;
      delete interop_ptr;
      //SHK *interop_ptr = omp_interop_none;
      interop_ptr = omp_interop_none;
      printf("__kmpc_interop_init(): Step1 setting *interop_ptr = omp_interop_none \n");
    }
  } else {
    if (!Device.RTL || !Device.RTL->init_device_info ||
        //SHK Device.RTL->init_device_info(device_id, &(*interop_ptr)->device_info,
        //                             &(*interop_ptr)->err_str)) {
        Device.RTL->init_device_info(device_id, &(interop_ptr)->device_info,
                                     &(interop_ptr)->err_str)) {
      // error
      //SHK delete *interop_ptr;
      delete interop_ptr;
      //SHK *interop_ptr = omp_interop_none;
      printf("__kmpc_interop_init(): Step2 setting *interop_ptr = omp_interop_none \n");
    }
  }
}

//EXTERN 
/*void __kmpc_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                               omp_interop_val_t **interop_ptr,
                               kmp_interop_type_t interop_type,
                               kmp_int32 device_id, kmp_int32 ndeps,
                               kmp_depend_info_t *dep_list,
                               kmp_int32 ndeps_noalias,
                               kmp_depend_info_t *noalias_dep_list) {*/
void __kmpc_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t *interop_ptr,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list) {
  printf("In __kmpc_interop_use ndeps %d dep_list %i %i\n", ndeps,dep_list, *dep_list);
  printf("In __kmpc_interop_use ndeps %d dep_list %i \n", ndeps,&dep_list[0]);
  printf("In __kmpc_interop_use ndeps %d dep_list %i \n", ndeps,&dep_list[1]);
  for(int i = 0; i < ndeps; i++)
    printf("Dep %d %i %u %u %u %u \n", i, dep_list[i].base_addr, 
			  dep_list[i].len, 
			  dep_list[i].flags.in,
			  dep_list[i].flags.out,
			  dep_list[i].flags.mtx);
/*  typedef struct kmp_depend_info {
  intptr_t base_addr;
  size_t len;
  struct {
   unsigned in : 1;
   unsigned out : 1;
   unsigned mtx : 1;
  } flags;
} kmp_depend_info_t;
*/
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = interop_ptr;
  assert(interop_val != omp_interop_none &&
         "Cannot use uninitialized interop_ptr!");
  //assert((interop_type == kmp_interop_type_unknown ||
  //        interop_val->interop_type == interop_type) &&
  //       "Inconsistent interop_ptr-type usage!");
  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
//#ifdef SYNC_INTEROP
#if 1
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
    kmp_tasking_flags flags;
    //flags.tiedness = 1;
    //flags.proxy = 1;
    //flags.task_serial = 1;
    //flags.reserved = 42;
    flags.target = 1;
    flags.unshackled = 1;
    /*kmp_int32 flags32 = *((kmp_int32*)&flags);

    DeviceTy &Device = PM->Devices[device_id];
    kmp_task_t *task = __kmpc_omp_task_alloc(loc_ref, gtid, flags32,
                                            40,
                                            16,
                                            &__fake_task_use,
                                            device_id);
    //printf("Dev %p: T: %p\n", &Device, task);
    ((DeviceTy**)task->shareds)[0] = &Device;
    ((__tgt_async_info**)task->shareds)[1] = interop_val->async_info;
    assert(((DeviceTy**)task->shareds)[0] == &Device);

    int r = __kmpc_omp_task_with_deps(loc_ref, gtid,
                                    task, ndeps,
                                    dep_list,
                                    ndeps_noalias,
                                    noalias_dep_list);
    //printf("Done with fake task: %i \n", r);
    */
#endif
  }
}

//EXTERN 
/*void __kmpc_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t **interop_ptr,
                            kmp_interop_type_t interop_type,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list,
                            kmp_int32 ndeps_noalias,
                            kmp_depend_info_t *noalias_dep_list) {*/
void __kmpc_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t *&interop_ptr,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list) {
  printf("In __kmpc_interop_destroy \n");
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;	
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = interop_ptr;
  // Gracefully handle the destruction of none objects, I guess.
  if (interop_val == omp_interop_none)
    return;

  //assert((interop_type == kmp_interop_type_unknown ||
  //        interop_val->interop_type == interop_type) &&
  //       "Inconsistent interop_ptr-type usage!");
  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
//#ifdef SYNC_INTEROP
#if 1
     __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
#else
    kmp_tasking_flags flags;
    //flags.tiedness = 1;
    //flags.proxy = 1;
    //flags.task_serial = 1;
    //flags.reserved = 42;
    flags.target = 1;
    flags.unshackled = 1;
    kmp_int32 flags32 = *((kmp_int32*)&flags);
/*
    DeviceTy &Device = PM->Devices[device_id];
    kmp_task_t *task  =__kmpc_omp_task_alloc(loc_ref, gtid, flags32,
                                            40,
                                            16,
                                            &__fake_task_use,
                                            device_id);
    //printf("Dev %p: T: %p\n", &Device, task);
    ((DeviceTy**)task->shareds)[0] = &Device;
    ((__tgt_async_info**)task->shareds)[1] = interop_val->async_info;
    assert(((DeviceTy**)task->shareds)[0] == &Device);

    int r = __kmpc_omp_task_with_deps(loc_ref, gtid,
                                    task, ndeps,
                                    dep_list,
                                    ndeps_noalias,
                                    noalias_dep_list);
    //printf("Done with fake task: %i \n", r);
*/
#endif
  }

  delete interop_ptr;
  interop_ptr = omp_interop_none;
}
#ifdef __cplusplus
} //extern "C"
#endif
