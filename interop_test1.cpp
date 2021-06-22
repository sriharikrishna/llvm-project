//#include <omp.h>
#include "build/runtimes/runtimes-bins/openmp/runtime/src/omp.h"
#include "openmp/libomptarget/include/omptarget.h"
#include "openmp/libomptarget/include/interop.h"

int main(){
  float arr[1000];
  //async_openmp_work(arr);
  //omp_interop_val_t *o =NULL;
  omp_interop_t o = 0; 
  //intptr_t type;
  int err = 0;
#pragma omp interop init(targetsync: o) depend(inout: arr)
  //auto type = omp_get_interop_property_int(o, omp_ipr_fr_id);
   long int li = omp_get_interop_int(o, omp_ipr_fr_id, &err);
   printf("%15li [E:%3i] ", li, err);
  /*if (type == omp_ifr_cuda) {
    cudaStream_t s = omp_get_interop_property_ptr(o, omp_ipr_targetsync);
    cublasSetStream(s);
    call_cublas_async_stuff(arr);
   } else {
    // handle other cases
   }*/
#pragma omp interop destroy(o) depend(inout: arr)

      	return 0;
}
