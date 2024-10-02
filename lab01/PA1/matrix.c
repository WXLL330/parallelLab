#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 2000
#define THREAD_NUMS 8

int main(int argc, char *argv) {
    int i, j, k, i_, j_, k_;
    double sum;
    double sstart, send, stime, pstart, pend, ptime;
    double **A, **B, **C, **D;
    char flag = 0;

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
            C[i][j] = j-i*2;
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
    //     for (k = 0; k < N; k++) {
    //         double tempA = A[i][k];
    //         // sum = 0;
    //         for (j=0; j < N; j++) {
    //             C[i][j] +=  tempA * B[k][j];
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

    pstart = omp_get_wtime(); //start time measurement
        
    # pragma omp parallel num_threads(THREAD_NUMS)\
        shared(A, B, D) private(i, j, k, sum)
    {
        # pragma omp for collapse(2)
        for (i = 0; i < N; i++) {
            for (k = 0; k < N; k++) {
                double tempA = A[i][k];
                sum = 0;
                for (j=0; j < N; j++) {
                    D[i][j] +=  tempA * B[k][j];
                }
            }
        } 
    }
    
    /*
    for (i = 0; i < N; i+=10) {
        for (j = 0; j < N; j+=10) {
            for (k=0; k < N; k+=10) {
                for (i_=i; i_ < i+10 && i_ < N; i_++) {
                     for (j_=j; j_ < j+10 && j_ < N; j_++) {
                          sum = 0;
                          for (k_=k; k_ < k+10 && k_ < N; k_++) {
                               sum += A[i_][k_]*B[k_][j_];
                          }
                          D[i_][j_] += sum;
                     }
                }
                
                // printf("Thread number is %d\n", omp_get_num_threads());
            }
        }
    }
    */
    
    
    pend = omp_get_wtime(); //end time measurement
    ptime = pend-pstart;
    
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
    
    
    return(0);
}
