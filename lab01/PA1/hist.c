#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 16

void serial_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }
    /* Counting */
    int idx;

    {
        {
            for (i = 0; i < n; i++) {
                int val = (int)array[i];
                if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
                    idx = num_bins - 1;
                } else {
                    idx = val % num_bins;
                }
                bins[idx]++;
            }
        }
    }
    
}

void parallel_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }
    /* Counting */
    int idx;
    int local_n = n / NUM_THREADS + 1;

    # pragma omp parallel num_threads(NUM_THREADS)\
        shared(bins, local_n, array, n, num_bins) private(idx, i)
    {
        
        int thread_rank = omp_get_thread_num();
        int start = thread_rank*local_n;
        int end_ = start+local_n;
        int end = n>end_?end_:n;

        int *local_bins = (int*)malloc(num_bins * sizeof(int));
        for (i = 0; i < num_bins; i++) {
            local_bins[i] = 0;
        }

        for (i = start; i < end; i++) {
            int val = (int)array[i];
            if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
                idx = num_bins - 1;
            } else {
                idx = val % num_bins;
            }
            local_bins[idx]++;
        }

        # pragma omp critical
        for(i = 0; i < num_bins; i++){
            // # pragma omp atomic
            bins[i] += local_bins[i];
        }
        free(local_bins);
    } 
}


void parallel_histogram_task(float *array, int n, int *bins, int num_bins)
{
    if(n <= NUM_THREADS){
        return serial_histogram(array, n, bins, num_bins);
    }

    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }

    /* Counting */
    int idx;
    # pragma omp parallel num_threads(NUM_THREADS) \
        shared(bins, array, num_bins)
    {
        # pragma omp single
        {
            int local_n = n / NUM_THREADS + 1;

            for(i = 0; i < n; i+=local_n){
                int start = i;
                int end_ = start+local_n;
                int end = n>end_?end_:n;

                # pragma omp task \
                    shared(array, bins) firstprivate(start, end) private(i, idx)
                {
                    int *local_bins = (int*)calloc(num_bins, sizeof(int));    //每个线程都有自己的bin

                    for(i = start; i < end; i++){
                        int val = (int)array[i];
                        if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
                            idx = num_bins - 1;
                        } else {
                            idx = val % num_bins;
                        }
                        local_bins[idx]++;
                    }

                    # pragma omp critical
                    {
                        for(i = 0; i < num_bins; i++){
                            bins[i] += local_bins[i];
                        }
                    }
                    
                    free(local_bins);
                }
            }
            // # pragma omp taskwait
        }

    }
    
}

void generate_random_numbers(float *array, int n) 
{
    int i;
    float a = 10.0;
    for(i=0; i<n; ++i)
        array[i] = ((float)rand()/(float)(RAND_MAX)) * a;
}

int main(int argc, char* argv[])
{    
    int n;
    int num_bins = 10;
    float *array;
    int *bins, *pbins;

    FILE *input = fopen("input2.txt", "r");
    if (input == NULL){
        printf("Error opening input file.\n");
        return EXIT_FAILURE;
    }
    fscanf(input, "%d", &n);
    fclose(input);

    array = (float *)malloc(sizeof(float) * n);
    bins = (int*)malloc(sizeof(int) * num_bins);
    pbins = (int*)malloc(sizeof(int) * num_bins);
    generate_random_numbers(array, n);

    double sstart = omp_get_wtime();
    serial_histogram(array, n, bins, num_bins);
    double send = omp_get_wtime();

    double pstart = omp_get_wtime();
    // parallel_histogram_task(array, n, pbins, num_bins);
    parallel_histogram(array, n, pbins, num_bins);
    double pend = omp_get_wtime();
    
    int i;
    
    printf("Serial Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, bins[i]);
    }

    printf("Parallel Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, pbins[i]);
    }

    double stime = send-sstart;
    double ptime = pend-pstart;
    printf("Serial Running time: %f seconds\n", stime);
    printf("Parallel Running time: %f seconds\n", ptime);
    printf("speedup: %f\n", stime/ptime);

    int count=0, flag=0;
    for (i = 0; i < num_bins; i++) {
        if(bins[i] != pbins[i]){
            flag = 1;
        }
        count+=bins[i];
    }
    if(flag == 1 || count != n){
        printf("compute error\n");
    }else{
        printf("compute correct\n");
    }

    free(array);
    free(bins);
    free(pbins);

    return 0;
}

