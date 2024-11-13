#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

struct mystruct{
    int n;
    char s[3];
};

void read_from_file(const char *s, int *n, char *op)
{
    FILE *fp; 
    if ((fp = fopen(s, "r")) == NULL)
    {
        printf("Unable to open %s for reading.\n", s);
        exit(0);
    }
    fscanf(fp, "%d,%s", n, op);

    fclose(fp);
} 

void gen_vector(int size, double *vec, int seed)
{
    int i;
    srand(seed);
    for (i = 0; i < size; i++)
            vec[i] = 0 + 1.0*rand()/RAND_MAX * (1 - 0);
} 

// numprocs - 1 次之后每个进程就有了1个allreduce之后的元素
// 再 numprocs - 1 次之后每个进程就有了完全allreduce之后的向量
void RING_Allreduce(double *sendbuf, double *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int numprocs, id, acc = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Request send_request, recv_request;
    for(int i = 0; i < count; ++i){
        recvbuf[i] = sendbuf[i];
    }

    int snum = count / numprocs;   
    int snum_last = count - snum*(numprocs - 1), slen = 0, rlen = 0;
    int times = (numprocs - 1);
    int dest = (id + 1) % numprocs; // 后一个
    int src = abs((id - 1 + numprocs) % numprocs); // 前一个
    int sendoffset = id * snum, recvoffset = src * snum;
    for(int i = 0; i < times; ++i){
        slen = (sendoffset != (numprocs - 1)*snum) ? snum : snum_last;
        rlen = (recvoffset != (numprocs - 1)*snum) ? snum : snum_last;

        MPI_Sendrecv(recvbuf+sendoffset, slen, datatype, dest, 0, 
                    recvbuf+recvoffset, rlen, datatype, src, 0, comm, MPI_STATUS_IGNORE);

        for(int j = 0; j < rlen; ++j){
            if(op == MPI_SUM){
                recvbuf[recvoffset + j] += sendbuf[recvoffset + j];
            }
            else if(op == MPI_MAX){
                recvbuf[recvoffset + j] = sendbuf[recvoffset + j] > recvbuf[recvoffset + j]? sendbuf[recvoffset + j] : recvbuf[recvoffset + j];
            }
        }

        sendoffset = recvoffset;
        recvoffset = ((recvoffset - snum) < 0) ? (numprocs - 1)*snum : recvoffset - snum;
    }
    
    for(int i = 0; i < times; ++i){
        slen = (sendoffset != (numprocs - 1)*snum) ? snum : snum_last;
        rlen = (recvoffset != (numprocs - 1)*snum) ? snum : snum_last;
        
        MPI_Sendrecv(recvbuf+sendoffset, slen, datatype, dest, 0, 
                    recvbuf+recvoffset, rlen, datatype, src, 0, comm, MPI_STATUS_IGNORE);

        sendoffset = recvoffset;
        recvoffset = ((recvoffset - snum) < 0) ? (numprocs - 1)*snum : recvoffset - snum;
    }
    
    return;
}

int main(int argc, char **argv)
{
    int n = 0, myid, numprocs;
    char s[3];
    double *v, *x, *y, rstart, rend, nstart, nend, rtime, ntime;
    struct mystruct mydata;

    MPI_Op op;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    MPI_Datatype mytype;
    MPI_Datatype type[2] = {MPI_INT, MPI_CHAR};
    int blocklen[2] = {1, 3};
    MPI_Aint disp[] = {
            offsetof(struct mystruct, n),
            offsetof(struct mystruct, s)};
    MPI_Type_create_struct(2, blocklen, disp, type, &mytype);
    MPI_Type_commit(&mytype);

    if (myid == 0)
    {
        read_from_file("input2.txt", &(mydata.n), mydata.s);
        // n = mydata.n;
        // strcpy(s, mydata.s);
    }

    if (numprocs == 1) {
        v = (double *)malloc(mydata.n * sizeof(double));
        gen_vector(mydata.n, v, myid);
        x = (double *)calloc(mydata.n, sizeof(double));
        y = (double *)calloc(mydata.n, sizeof(double));
    }
    else{
        MPI_Bcast(&mydata, 1, mytype, 0, MPI_COMM_WORLD);
        n = mydata.n;
        strcpy(s, mydata.s);
        //printf("proc%d get n=%d, s:%s\n", myid, n, s);

        v = (double *)malloc(n * sizeof(double));
        x = (double *)calloc(n, sizeof(double));
        y = (double *)calloc(n, sizeof(double));
        gen_vector(n, v, myid+1);
        
        // printf("process%d v: ", myid);
        // for(int i = 0; i < n; ++i){
        //     printf("%f ", v[i]);
        // }
        // printf("\n");

        op = strcmp(s, "sum")==0 ? MPI_SUM : MPI_MAX;

        MPI_Barrier(MPI_COMM_WORLD);
        rstart = MPI_Wtime();
        RING_Allreduce(v, x, n, MPI_DOUBLE, op, MPI_COMM_WORLD);
        //MPI_Barrier(MPI_COMM_WORLD);
        rend = MPI_Wtime();
        rtime = rend - rstart;

        MPI_Barrier(MPI_COMM_WORLD);
        nstart = MPI_Wtime();
        MPI_Allreduce(v, y, n, MPI_DOUBLE, op, MPI_COMM_WORLD);
        // MPI_Barrier(MPI_COMM_WORLD);
        nend = MPI_Wtime();
        ntime = nend - nstart;
    }

    if(myid == 0){
        printf("Ring based Allreduce runtime: %f\n", rtime);
        printf("Normal Allreduce runtime: %f\n", ntime);
        printf("Speedup: %f\n", ntime / rtime);

        printf("ring: ");
        for(int i = 0; i < n; ++i){
            printf("%lf ", x[i]);
        }
        printf("\n");
        printf("norm: ");
        for(int i = 0; i < n; ++i){
            printf("%lf ", y[i]);
        }
        printf("\n");

        for(int i = 0; i < n; ++i){
            if(abs(x[i] - y[i]) > 5e-5){
                printf("compute error!\n");
                exit(1);
            }
        }
        printf("compute correct!\n");
    }
    
    free(v);
    free(x);
    free(y);
    MPI_Type_free(&mytype);
    MPI_Finalize();
    return 0;
}
