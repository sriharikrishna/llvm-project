#define CN (1<<11)

#if 1
#define VERBOSE(...) printf(__VA_ARGS__)
#else
#define VERBOSE(...)
#endif


#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

__attribute__((noinline))
void work(double *A, double *B, int N) {
  for (int i =0; i < N; ++i)
    B[i] = i;
  for (int i =0; i < N; ++i)
    for (int j =0; j < N; ++j)
      A[i] += B[j];
}

void busy_work() {
      double *A = (double*)malloc(CN * sizeof(A[0]));
      double *B = (double*)malloc(CN * sizeof(B[0]));
      work(A, B, CN);
      free(A);
      free(B);
}

typedef void* cudaStream_t;

extern "C" {
int cudaMalloc(void **, size_t);
int cudaMemcpy(void *dst, const void *src, size_t count, int kind);
int cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, cudaStream_t stream );
int cudaStreamSynchronize(cudaStream_t stream);
}

int main() {

  /*
   * R0 = cudaMalloc();
   * R1 = cudaMalloc();
   * R2 = 0;
   *
   * Task OMP T0       depend(out:D0) is_device_ptr(R0) nowait
   *   some busy work
   *   R0 = 1;
   *
   *
   * interop init
   *
   * interop use T1    depend( in:D0) depend(out:D2)
   *
   * cudaMemcpyAsync(&R1, &R0, ...)
   *
   * interop destroy   depend( in:D0) depend(out:D2)
   *
   * Task OMP T2                      depend( in:D2) is_device_ptr(R1) map(from:R2) nowait
   *   R2 = R1;
   *
   *
   * After all tasks finished R2 should be 1
   *
   */

  printf("Welcome :)\n#devices %i, device: %i, initial device: %i, default device: %i\n", omp_get_num_devices(), omp_get_device_num(), omp_get_initial_device(), omp_get_default_device());

  int device_id = omp_get_default_device();
  printf("main(): device_id %d\n", device_id);
  int *R0, *R1, R2 = 0;
  cudaMalloc((void**)&R0, sizeof(int));
  cudaMalloc((void**)&R1, sizeof(int));

  printf("[%lf] Start the parallel master\n", omp_get_wtime());

  #pragma omp parallel master
  {

    double D0, D2;
    #pragma omp target depend(out:D0) is_device_ptr(R0) nowait
    {
      busy_work();
      busy_work();
      *R0 = 1;
      VERBOSE("[%lf] End OMP T0\n", omp_get_wtime());
    }
    VERBOSE("[%lf] After spawn OMP T0: R2 %i\n", omp_get_wtime(), R2);
      
    omp_interop_t interop;
    #pragma omp interop init(targetsync: interop) device(device_id)
    
    int err;
    for (int i = omp_ipr_first; i < 0; i++) {
      VERBOSE("%i: ", i);
      const char *n = omp_get_interop_name(interop, (omp_interop_property_t)(i));
      VERBOSE("%15s ", n);
      long int li = omp_get_interop_int(interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15li [E:%3i] ", li, err);
      const void *p = omp_get_interop_ptr(interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15p [E:%3i] ", p, err);
      const char *s = omp_get_interop_str(interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15s [E:%3i] ", s, err);
      const char *n1 = omp_get_interop_type_desc(interop, (omp_interop_property_t)(i));
      VERBOSE("%15s \n", n1);
    }
    VERBOSE("After interop inspect: R2 %i\n", R2);
    #pragma omp interop use(interop) depend(in:D0, D2)  
    fflush(stdout);
    VERBOSE("[%lf] After interop use: %i\n", omp_get_wtime(), R2);
    
    cudaStream_t stream = (omp_get_interop_ptr(interop, omp_ipr_targetsync, NULL));
    VERBOSE("After get stream %i: R2 %i\n", stream, R2);

    int cudaMemcpyDefault = 4;
    VERBOSE("Issue CUDA async memcpy       on stream (=%i)\n", stream);
    int r = cudaMemcpyAsync(&R1, &R0, sizeof(int), cudaMemcpyDefault, stream);
    VERBOSE("[%lf] After cuda memcpy async [%i]: R2 %i\n", omp_get_wtime(), r, R2);

    #pragma omp interop destroy(interop) depend(in:D0, D2)   
    VERBOSE("[%lf] After interop destroy: R2 %i\n", omp_get_wtime(), R2);

    #pragma omp target depend(in:D2) is_device_ptr(R0,R1) map(from:R2) nowait
    {
      R2 = *R1;
      VERBOSE("[%lf] End OMP T2 *R0 %i R2 %i\n", omp_get_wtime(), *R0, R2);
    }
    VERBOSE("[%lf] After spawn OMP T2: R2 %i\n", omp_get_wtime(), R2);

    printf("\n[%lf] - Setup done, waiting for T0 to finish, to trigger T1 which triggers T2, result: R2 %i (probably still 0)\n",omp_get_wtime(), R2);

    busy_work();
    busy_work();
    printf("\n[%lf] - After first  wait: result: R2 %i\n", omp_get_wtime(), R2);

    busy_work();
    busy_work();
    printf("\n[%lf] - After second wait: result: R2 %i - Waiting at the parallel region's (implicit) barrier now\n", omp_get_wtime(), R2);

    busy_work();
    busy_work();
    printf("\n[%lf] - After third  wait: result: R2 %i - Waiting at the parallel region's (implicit) barrier now\n", omp_get_wtime(), R2);
  }
  
  printf("[%lf] Final: %i (should be 1)\n", omp_get_wtime(), R2);
  assert(R2 == 1);
}
