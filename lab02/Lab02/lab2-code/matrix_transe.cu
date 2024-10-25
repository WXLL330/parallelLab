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


const int TILE_WIDTH = 16;
__global__ void MatrixTranseKernel(int *d_M, int *d_P, int m, int n)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //Identify the row and column of the Pd element to work on
  int col = bx * TILE_WIDTH + tx;
  int row = by * TILE_WIDTH + ty;

  //loop over the Md and Nd tiles required to compute the Pd element
  if(row < m && col < n){
    d_P[col*m+row] = d_M[row*n+col];
  }
}

__global__ void MatrixTranseKernelShared(int *d_M, int *d_P, int m, int n)
{
  __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH+1];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //Identify the row and column of the Pd element to work on
  int col = bx * TILE_WIDTH + tx;
  int row = by * TILE_WIDTH + ty;
  // int total = m*n;

  //loop over the Md and Nd tiles required to compute the Pd element
  // if(row < m && col < n){
  //   ds_M[ty][tx] = d_M[row*n+col];
  // }
  
  ds_M[ty][tx] = (row < m && col < n) * d_M[row*n + col];
  __syncthreads();

  // if(row < m && col < n){
  //   printf("%d ", ds_M[row][col]);
  // }

  // 这里对d_P的访问是聚合的
  row = bx * TILE_WIDTH + ty;
  col = by * TILE_WIDTH + tx;
  d_P[row*m+col] = (row < n && col < m) * ds_M[tx][ty];
  // if(row < n && col < m)
  //   d_P[row*m+col] = ds_M[tx][ty];

  // 这里对d_P的访问不是聚合的
  // row = bx * TILE_WIDTH + tx;
  // col = by * TILE_WIDTH + ty;
  // if(row < n && col < m)
  //   d_P[row*m+col] = ds_M[ty][tx];
}

// 1 0 1 0     



int main()
{
  //freopen("out","w",stdout);
  int m = 40000, n = 40000;
  int *h_M, *t_M, *d_M, *d_P, *h_P, *h_PS;

  size_t sizeM = m * n * sizeof(int);

  h_M = (int *) malloc(sizeM);
  t_M = (int *) malloc(sizeM);
  h_P = (int *) malloc(sizeM);
  h_PS = (int *) malloc(sizeM);

  cudaMalloc(&d_M,sizeM);
  cudaMalloc(&d_P,sizeM);

  for(int i = 0; i < m * n; ++i)
  {
    if(i % 2 == 0)
      h_M[i] = 1;
    else
      h_M[i] = 0;
  }

  cudaMemcpy(d_M,h_M,sizeM,cudaMemcpyHostToDevice);

  cudaEvent_t start,stop;
  float gElapsedTime, cElapsedTime;

  // GPU Transe without SharedMemory
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start,0));

  dim3 grid((int)ceil(n*1.0/ TILE_WIDTH), (int)ceil(m*1.0/ TILE_WIDTH));
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  MatrixTranseKernel<<<grid,block>>>(d_M, d_P, m, n);

  CHECK(cudaEventRecord(stop,0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&gElapsedTime,start,stop));
  printf("Kernel Elpased Time without SharedMemory: %.3f ms\n",gElapsedTime);

  CHECK(cudaMemcpy(h_P,d_P,sizeM,cudaMemcpyDeviceToHost));


  // GPU Transe with SharedMemory
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start,0));

  MatrixTranseKernelShared<<<grid,block>>>(d_M, d_P, m, n);

  CHECK(cudaEventRecord(stop,0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&gElapsedTime,start,stop));
  printf("Kernel Elpased Time with SharedMemory: %.3f ms\n",gElapsedTime);

  CHECK(cudaMemcpy(h_PS,d_P,sizeM,cudaMemcpyDeviceToHost));


  // CPU Transe
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start,0));
  
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      t_M[j*m + i] = h_M[i*n + j];
    }
  }

  CHECK(cudaEventRecord(stop,0));
  //cudaDeviceSynchronize();
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&cElapsedTime,start,stop));
  printf("CPU Elpased Time: %.3f ms\n",cElapsedTime);

  printf("SpeedUp:%f\n", cElapsedTime/gElapsedTime);

  // printf("h_PS:\n");
  // for(int i = 0; i < n; i++){
  //   for(int j = 0; j < m; j++){
  //     printf("%d ", h_PS[i*m + j]);
  //   }
  //   printf("\n");
  // }

  // printf("\nt_M:\n");
  // for(int i = 0; i < n; i++){
  //   for(int j = 0; j < m; j++){
  //     printf("%d ", t_M[i*m + j]);
  //   }
  //   printf("\n");
  // }
  
  for(int i = 0; i < n*m; ++i){
    if(t_M[i] != h_P[i]){
        printf("1 compute error\n");
        free(h_P);
        free(h_PS);
        free(h_M);
        free(t_M);
        cudaFree(d_P);
        cudaFree(d_M);
        return -1;
    }
  }
  for(int i = 0; i < n*m; ++i){
    if(t_M[i] != h_PS[i]){
        printf("2 compute error\n");
        free(h_P);
        free(h_PS);
        free(h_M);
        free(t_M);
        cudaFree(d_P);
        cudaFree(d_M);
        return -1;
    }
  }
  printf("coumpute correct\n");

 free(h_P);
 free(h_PS);
 free(h_M);
 free(t_M);
 cudaFree(d_P);
 cudaFree(d_M);

  return 0;
}
