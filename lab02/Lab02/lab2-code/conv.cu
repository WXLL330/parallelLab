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

const int TILE_WIDTH = 32;
__global__ void ConvKernel(int *d_M, int *d_K, int *d_N, int m, int n, int k)
{
    // __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
    // __shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int pValue = 0;
    int m_ = m - k + 1;
    int n_ = n - k + 1;

    // ds_M[ty][tx] = (row < m && col < n) * d_M[row*n + col];

    
    for(int i = 0; i < k; ++i){
        for(int j = 0; j < k; ++j){
            pValue += ((row+i)<m && (col+j)<n) * d_M[(row+i)*n + col + j] * d_K[i*k + j];
        }
    }
    
    if(row < m_ && col < n_)
        d_N[row*n_ + col] = pValue;



    //loop over the Md and Nd tiles required to comput the Pd element
    // for(int t = 0; t < (n-1) / TILE_WIDTH + 1; ++t)
    // {
    //     if(row < m && t * TILE_WIDTH + tx < n)
    //     ds_M[ty][tx] = d_M[row * n + t * TILE_WIDTH + tx];
    //     else
    //     ds_M[ty][tx] = 0;

    //     __syncthreads();

    //     for(int i = 0; i < TILE_WIDTH; ++i)
    //     pValue += ds_M[ty][i] * ds_N[i][tx];
    //     __syncthreads();
    // }
    // if(row < m && col < k)
    //     d_P[row * k + col] = pValue;
}

int main()
{
    //freopen("out","w",stdout);
    int m = 50, n = 90000, k = 5;
    int m_ = m - k + 1;
    int n_ = n - k + 1;

    int *h_M, *h_N, *h_K, *d_M, *d_N, *d_K;
    int *h_P;

    size_t sizeM = m * n * sizeof(int);
    size_t sizeN = m_ * n_ * sizeof(int);
    size_t sizeK = k * k * sizeof(int);

    h_M = (int *) malloc(sizeM);
    h_N = (int *) malloc(sizeN);
    h_P = (int *) malloc(sizeN);
    h_K = (int *) malloc(sizeK);

    CHECK(cudaMalloc(&d_M,sizeM));
    CHECK(cudaMalloc(&d_N,sizeN));
    CHECK(cudaMalloc(&d_K,sizeK));

    for(int i = 0; i < m * n; ++i)
    {
        if(i % 2 == 0)
            h_M[i] = 1;
        else
            h_M[i] = 0;
    }

    for(int i = 0; i < k * k; ++i){
        h_K[i] = 1;
    }
    
    CHECK(cudaMemcpy(d_M,h_M,sizeM,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K,h_K,sizeK,cudaMemcpyHostToDevice));

    cudaEvent_t start,stop;

    // CUDA Conv
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start,0));

    dim3 grid((int)ceil(n_*1.0 / TILE_WIDTH), (int)ceil(m_*1.0/ TILE_WIDTH));
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    ConvKernel<<<grid,block>>>(d_M, d_K, d_N, m, n, k);

    CHECK(cudaEventRecord(stop,0));
    //cudaDeviceSynchronize();
    CHECK(cudaEventSynchronize(stop));
    float gElapsedTime;
    CHECK(cudaEventElapsedTime(&gElapsedTime,start,stop));
    printf("Kernel Elpased Time: %.3f ms\n",gElapsedTime);

    cudaMemcpy(h_P,d_N,sizeN,cudaMemcpyDeviceToHost);


    // CPU Conv
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start,0));

    int sum = 0;
    for(int i = 0; i < m_; i++){
        for(int j = 0; j < n_; j++){
            for(int i_ = 0; i_ < k; i_++){
                for(int j_ = 0; j_ < k; j_++){
                    sum += h_M[(i+i_)*n + j+j_] * h_K[i_*k + j_];
                }
            }
            h_N[i*n_ + j] = sum;
            sum = 0;
        }
    }

    CHECK(cudaEventRecord(stop,0));
    //cudaDeviceSynchronize();
    CHECK(cudaEventSynchronize(stop));
    float cElapsedTime;
    CHECK(cudaEventElapsedTime(&cElapsedTime,start,stop));
    printf("CPU Elpased Time: %.3f ms\n",cElapsedTime);
    printf("SpeedUp: %f\n", cElapsedTime/gElapsedTime);

    // printf("h_M: \n");
    // for(int i = 0; i < m; ++i){
    //     for(int j = 0; j < n; ++j){
    //         printf("%d ", h_M[i*n + j]);
    //     }
    //     printf("\n");
    // }
    // printf("h_N: \n");
    // for(int i = 0; i < m_; ++i){
    //     for(int j = 0; j < n_; ++j){
    //         printf("%d ", h_N[i*n_ + j]);
    //     }
    //     printf("\n");
    // }
    // printf("h_P: \n");
    // for(int i = 0; i < m_; ++i){
    //     for(int j = 0; j < n_; ++j){
    //         printf("%d ", h_P[i*n_ + j]);
    //     }
    //     printf("\n");
    // }

    for(int i = 0; i < m_*n_; ++i){
        if(h_N[i] != h_P[i]){
            printf("compute error, i: %d, cpu: %d, gpu: %d\n", i, h_N[i], h_P[i]);
            free(h_P);
            free(h_K);
            free(h_M);
            free(h_N);
            CHECK(cudaFree(d_K));
            CHECK(cudaFree(d_M));
            CHECK(cudaFree(d_N));
            return -1;
        }
    }
    printf("compute correct\n");


    free(h_P);
    free(h_K);
    free(h_M);
    free(h_N);
    CHECK(cudaFree(d_K));
    CHECK(cudaFree(d_M));
    CHECK(cudaFree(d_N));

    return 0;
}
