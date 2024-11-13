#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "time.h"


struct mystruct {
    int m;
    int k;
    int n;
};

void mat_mul(double *A, double *B, double *C, int m, int k, int n)
{
    double temp;
    int i, j, t;
    for (i = 0; i < m; i++){
        for (t = 0; t < k; t++){
        	temp = A[i*k + t];
        	for(j = 0; j < n; j++){
        		C[i*n + j] += temp * B[t*n + j];
        	}
        }
    }
}

void gen_matrix(int row, int col, double *mat)
{
    int i;
    srand(time(0));
    for (i = 0; i < row * col; i++)
            mat[i] = 0 + 1.0*rand()/RAND_MAX * (1 - 0);
} 

void print_matrix(int row, int col, double *mat)
{
    for(int j = 0; j < row; j++){
        for(int k = 0; k < col; k++){
            printf("%f ", mat[j*col+k]);
        }
        printf("\n");
    }
    printf("\n");
}

void read_from_file(const char *s, int *m, int *k, int *n)
{
    FILE *fp; 
    if ((fp = fopen(s, "r")) == NULL)
    {
        printf("Unable to open %s for reading.\n", s);
        exit(0);
    }
    fscanf(fp, "%d,%d,%d", m, k, n);

    fclose(fp);
} 

int main(int argc, char **argv)
{
    int m = 0, n = 0, k = 0, myid, numprocs, srow = 0, i, srow_last = 0;
    double *A, *B, *C, *D, start, end;
    struct mystruct mydata;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    // printf("total size is %d\n", numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    // printf("I am proc %d\n", myid);

    // create my datatype
    MPI_Datatype mytype;
    MPI_Datatype type[3] = {MPI_INT, MPI_INT, MPI_INT};
    int blocklen[3] = {1, 1, 1};
    MPI_Aint disp[] = {offsetof(struct mystruct, m),
              offsetof(struct mystruct, k),
              offsetof(struct mystruct, n)};
    MPI_Type_create_struct(3, blocklen, disp, type, &mytype);
    MPI_Type_commit(&mytype);

    if (myid == 0)
    {
        read_from_file("input1.txt", &(mydata.m), &(mydata.k), &(mydata.n));
        m = mydata.m;
        k = mydata.k;
        n = mydata.n;
        A = (double *)malloc(m * k * sizeof(double));
        B = (double *)malloc(k * n * sizeof(double));
        gen_matrix(m, k, A);
        gen_matrix(k, n, B);

        start = MPI_Wtime();
    }
    
    if (numprocs == 1) {
        C = (double *)malloc(m * n * sizeof(double));
        mat_mul(A, B, C, m, k, n);   
    }
    else{
        // MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mydata, 1, mytype, 0, MPI_COMM_WORLD);
        if (myid != 0){
            m = mydata.m;
            k = mydata.k;
            n = mydata.n;
            B = (double *)malloc(k * n * sizeof(double));
        }
        printf("process %d gets n = %d, m = %d, k = %d\n", myid, n, m, k);

        /* broadcast matrix B */
        MPI_Bcast(B, k*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 计算每个进程需要计算的行数
        srow = m / numprocs;
        srow_last = m - srow * (numprocs - 1);
        if (myid == numprocs - 1)
            srow = srow_last;
        printf("process %d calculates %d rows\n", myid, srow);
        if (myid == 0)
        {
            /* master code */
            C = (double *)malloc(m * n * sizeof(double));

            /* send sub-matrices to other processes */
            for (i = 1; i < numprocs - 1; i++)
                MPI_Send(A + i * srow * k, srow * k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(A + i * srow * k, srow_last * k, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD);

            /* perform its own calculation for the 1st sub-matrix */
            mat_mul(A, B, C, srow, k, n);

            /* collect results from other processes */
            for (i = 1; i < numprocs - 1; i++)
                MPI_Recv(C + i * srow * n, srow*n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(C + i * srow * n, srow_last*n, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
            printf("\n");
        }
        else
        {
            /* slave code */
            /* allocate memory for sub-matrix A, and sub-sector y */
            A = (double *)malloc(srow * k * sizeof(double));
            C = (double *)malloc(srow * n * sizeof(double));

            /* receive sub-matrix from process 0 */
            MPI_Recv(A, srow * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* perform the calculation on the sub-matrix */
            mat_mul(A, B, C, srow, k, n);

            /* send the results to process 0 */
            MPI_Send(C, srow*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        // MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if(myid == 0)
    {
        end = MPI_Wtime();
    	double ptime = end - start;
    	printf("parallel runtime=%lf\n", ptime);
    	
    	D = (double *)malloc(m * n * sizeof(double));
    	if(D == NULL)
    	{
    		printf("malloc for D fail\n");
    		exit(1);
    	}
 
    	start = MPI_Wtime();
    	mat_mul(A, B, D, m, k, n);
    	end = MPI_Wtime();
    	double stime = end - start;
    	printf("serial runtime=%lf\n", stime);
    	printf("SpeedUp: %lf\n", stime/ptime);
    	for(int j = 0; j < m*n; j++)
    	{
    		if(abs(C[j]-D[j]) > 5e-5)
    		{
	    		printf("compute error\n");
	    		exit(1);
    		}
    	}
    	printf("compute correct\n");

        FILE *fp;
        if ((fp = fopen("output1.txt", "w")) == NULL)
        {
            printf("Unable to open output1.txt for writing.\n");
            exit(1);
        }
        fprintf(fp, "%.2f,%.2f", ptime, stime);

        fclose(fp);
    }
    else
    {
    	D = (double *)malloc(m * n * sizeof(double));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    free(A);
    free(B);
    free(C);
    free(D);
    MPI_Finalize();
    return 0;
}
