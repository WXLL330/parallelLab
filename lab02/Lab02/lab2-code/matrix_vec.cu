#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
using namespace std;

#define CHECK(call)                                                    \
    do {                                                               \
        const cudaError_t error = call;                                \
        if (error != cudaSuccess) {                                    \
            printf("Error: %s:%d, ", __FILE__, __LINE__);              \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                   \
        }                                                              \
    } while (0)


const int TILE_WIDTH = 128;
__global__ void MatrixMulKernel(int *d_M,int *d_V,int *d_P,int m,int n)
{
  __shared__ int ds_V[TILE_WIDTH];

  int bx = blockIdx.x;

  int tx = threadIdx.x;

  int idx = bx * TILE_WIDTH + tx;

  int pValue = 0;

  for(int i = 0; i < (n-1)/TILE_WIDTH+1; ++i){
    int pos = i*TILE_WIDTH + tx;
    ds_V[tx] = pos<n?d_V[pos]:0;
    __syncthreads();
    
    if(idx < m){
      for(int j = 0; j < TILE_WIDTH; j++){
        pValue += i*TILE_WIDTH+j < n?d_M[idx*n + i*TILE_WIDTH+j] * ds_V[j]:0;
      }
    }
    
    __syncthreads();
  }
  
  d_P[idx] = pValue;
}


int main()
{
  //freopen("out","w",stdout);
  int m = 20000, n = 50000;
  int *h_M, *h_V, *d_M, *d_V;
  int *h_P, *d_P, *P;
  size_t sizeM = m * n * sizeof(int);
  size_t sizeV = n * sizeof(int);
  size_t sizeP = m * sizeof(int);

  h_M = (int *) malloc(sizeM);
  // T_M = (int *) malloc(sizeM);
  h_V = (int *) malloc(sizeV);
  h_P = (int *) malloc(sizeP);
  P = (int *) malloc(sizeP);
  
  if(h_M == NULL || h_V == NULL || h_P == NULL || P == NULL){
    printf("cpu memory malloc error\n");
    exit(1);
  }

  CHECK(cudaMalloc(&d_M, sizeM));
  CHECK(cudaMalloc(&d_V, sizeV));
  CHECK(cudaMalloc(&d_P, sizeP));

  for(int i = 0; i < m * n; ++i)
  {
    if(i % 2 == 0)
      h_M[i] = 1;
    else
      h_M[i] = 0;
  }

  for(int i = 0;i < n; ++i)
  {
    if(i % 2 == 0)
      h_V[i] = 0;
    else
      h_V[i] = 1;
  }

  CHECK(cudaMemcpy(d_M,h_M,sizeM,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_V,h_V,sizeV,cudaMemcpyHostToDevice));

  cudaEvent_t start,stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start,0));

  dim3 grid((m-1)/TILE_WIDTH + 1);
  dim3 block(TILE_WIDTH);
  MatrixMulKernel<<<grid,block>>>(d_M,d_V,d_P,m,n);

  CHECK(cudaEventRecord(stop,0));
  //cudaDeviceSynchronize();
  CHECK(cudaEventSynchronize(stop));
  float gElapsedTime, cElapsedTime;
  CHECK(cudaEventElapsedTime(&gElapsedTime,start,stop));
  printf("Kernel Elpased Time: %.3f ms\n",gElapsedTime);
  
  
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start,0));
  
  int pValue = 0;
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      pValue += h_M[i*n + j] * h_V[j];
    }
    P[i] = pValue;
  }

  CHECK(cudaEventRecord(stop,0));
  //cudaDeviceSynchronize();
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&cElapsedTime,start,stop));
  printf("CPU Elpased Time: %.3f ms\n",cElapsedTime);
  
  printf("SpeedUp:%f\n", cElapsedTime/gElapsedTime);
  
  CHECK(cudaMemcpy(h_P,d_P,sizeP,cudaMemcpyDeviceToHost));
  
  for(int i = 0; i < m; ++i){
    if(P[i] != h_P[i]){
       printf("compute error\n");
       return -1;
    }
  }
  printf("coumpute correct\n");

  free(h_P);
  free(h_M);
  // free(T_M);
  free(h_V);
  free(P);
  cudaFree(d_P);
  cudaFree(d_M);
  cudaFree(d_V);

  return 0;
}
