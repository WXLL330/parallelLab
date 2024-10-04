#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_NUMS 16
#define BLOCK_SIZE 128

int main(int argc, char *argv) {
    int i, j, k, i_, j_, k_, N;
    double sum;
    double sstart, send, stime, pstart, pend, ptime;
    double **A, **B, **C, **D;
    char flag = 0;

    FILE *input = fopen("input1.txt", "r");
    if (input == NULL){
        printf("Error opening input file.\n");
        return EXIT_FAILURE;
    }
    fscanf(input, "%d", &N);
    fclose(input);

    A = (double**)malloc(N * sizeof(double*));
    B = (double**)malloc(N * sizeof(double*));
    C = (double**)malloc(N * sizeof(double*));
    D = (double**)malloc(N * sizeof(double*));
    A[0] = (double*)malloc(N * N * sizeof(double));
    B[0] = (double*)malloc(N * N * sizeof(double));
    C[0] = (double*)malloc(N * N * sizeof(double));
    D[0] = (double*)malloc(N * N * sizeof(double));

    for (i = 1; i < N; i++) {
        A[i] = A[0] + i * N;
        B[i] = B[0] + i * N;
        C[i] = C[0] + i * N;
        D[i] = D[0] + i * N;
    }
    
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = j*1;
            B[i][j] = i*j+2;
            C[i][j] = 0;
            D[i][j] = 0;
        }
    }
    
    sstart = omp_get_wtime(); //start time measurement
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k=0; k < N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    // for (i = 0; i < N; i++) {
    //     double *tempA = A[i];
    //     for (k = 0; k < N; k++) { 
            
    //         for (j = 0; j < N; j++){
    //             C[i][j] +=  tempA[k] * B[k][j];
    //         }
    //     }
    // } 
    send = omp_get_wtime(); //end time measurement
    stime = send-sstart;
   
    
    /*
    // 
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            int a = A[i][k];
            sum = 0;
            for (j=0; j < N; j++) {
                C[i][j] += a*B[k][j];
            }
            //C[i][j] = sum;
        }
    }
    */

    omp_set_nested(1);  //开启嵌套并行
    int chunk_size = (N / THREAD_NUMS) / 2;
    pstart = omp_get_wtime(); //start time measurement

    // 空间局部性
    // # pragma omp parallel num_threads(THREAD_NUMS) \
    //     shared(A, B, D) private(i, j, k) 
    // {
    //     # pragma omp for schedule(static, chunk_size)
    //     for (i = 0; i < N; i++) {
    //         double *tempA = A[i];
    //         for (k = 0; k < N; k++) { 
    //             for (j = 0; j < N; j++){
    //                 D[i][j] +=  tempA[k] * B[k][j];
    //             }
    //         }
    //     } 
    // }

    
    {
        # pragma omp parallel for num_threads(4)  schedule(static, 256)\
            shared(A, B, D) private(i, j, k, sum)
        for (i = 0; i < N; i++) {
            # pragma omp parallel for num_threads(4) schedule(dynamic)\
                shared(A, B, D) private(j, k, sum)
            for (j = 0; j < N; j++) {
                sum = 0;
                for (k=0; k < N; k++) {
                    sum += A[i][k]*B[k][j];
                }
                D[i][j] = sum;
            }
        }
    }
    
    
    // 分块
    // # pragma omp parallel num_threads(THREAD_NUMS)\
    //     shared(A, B, D) private(i, j, k, sum, i_, j_, k_)
    // {
    //     # pragma omp for collapse(2) schedule(dynamic)
    //     for (i = 0; i < N; i+=BLOCK_SIZE) {
    //         for (j = 0; j < N; j+=BLOCK_SIZE) {
    //             for (k=0; k < N; k+=BLOCK_SIZE) {
    //                 for (i_=i; i_ < i+BLOCK_SIZE; i_++) {
    //                     for (j_=j; j_ < j+BLOCK_SIZE; j_++) {
    //                         sum = 0;
    //                         for (k_=k; k_ < k+BLOCK_SIZE; k_++) {
    //                             sum += A[i_][k_]*B[k_][j_];
    //                         }
    //                         D[i_][j_] += sum;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }



    // {
    //     # pragma omp parallel for num_threads(8)\
    //         shared(A, B, D) private(i, j, k, sum, i_, j_, k_)
    //     for (i = 0; i < N; i+=BLOCK_SIZE) {
    //         for (j = 0; j < N; j+=BLOCK_SIZE) {
    //             for (k=0; k < N; k+=BLOCK_SIZE) {
    //                 # pragma omp parallel for num_threads(2)\
    //                     shared(A, B, D) private(sum, i_, j_, k_)
    //                 for (i_=i; i_ < i+BLOCK_SIZE; i_++) {
    //                     for (j_=j; j_ < j+BLOCK_SIZE; j_++) {
    //                         sum = 0;
    //                         for (k_=k; k_ < k+BLOCK_SIZE; k_++) {
    //                             sum += A[i_][k_]*B[k_][j_];
    //                         }
    //                         D[i_][j_] += sum;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    
    
    
    pend = omp_get_wtime(); //end time measurement
    ptime = pend-pstart;

    FILE *output = fopen("output1.txt", "w");
    if (output == NULL){
        printf("Error opening output file.\n");
        return EXIT_FAILURE;
    }
    fprintf(output, "%.2f,%.2f\n", ptime*1000, stime*1000);
    fclose(output);
    
    printf("Time of serial computation: %f seconds\n", stime);
    printf("Time of parallel computation: %f seconds\n", ptime);
    printf("serial/parallel = %f\n", stime/ptime);
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if(C[i][j] != D[i][j]){
                flag = 1;
                break;
            }
        }
    }
    if(flag == 1){
        printf("parallel compute error\n");
    }
    free(A);
    free(B);
    free(C);
    free(D);
    
    
    return(0);
}
