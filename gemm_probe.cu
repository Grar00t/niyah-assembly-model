// /tmp/gemm_probe.cu
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); return 1; } } while(0)

int main(){
  int n=4096; size_t sz=(size_t)n*n*sizeof(float);
  float *A,*B,*C;
  CK(cudaMalloc(&A,sz)); CK(cudaMalloc(&B,sz)); CK(cudaMalloc(&C,sz));
  CK(cudaMemset(A,0,sz)); CK(cudaMemset(B,0,sz));

  cublasHandle_t h; CB(cublasCreate(&h));
  float alpha=1.f,beta=0.f;

  // warm-up
  for(int i=0;i<3;i++) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,n,B,n,&beta,C,n));
  CK(cudaDeviceSynchronize());

  cudaEvent_t s,e;
  CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));

  CK(cudaEventRecord(s,0));
  for(int i=0;i<20;i++) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,n,B,n,&beta,C,n));
  CK(cudaEventRecord(e,0));
  CK(cudaEventSynchronize(e));   // ← إلزامي

  float ms=0.f;
  CK(cudaEventElapsedTime(&ms,s,e));
  if (ms <= 0.f) { printf("invalid elapsed time: %.3f\n", ms); return 1; }

  double tflops = 2.0*n*n*n*20/(ms*1e-3)/1e12;
  printf("SGEMM %dx%d: %.3f ms/call, %.2f TFLOPS\n",n,n,ms/20.0,tflops);
  CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
  CB(cublasDestroy(h));
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
