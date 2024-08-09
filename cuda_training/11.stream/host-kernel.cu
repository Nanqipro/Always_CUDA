#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N - 1) / block_size + 1;

void cpu_sum(const real *h_x, const real *h_y, real *h_z, const int N){
    for(int i=0; i<N; i++){
        h_z[i] = h_x[i] + h_y[i];
    }
}

void __global__ gpu_sum(const real *d_x, const real *d_y, real *d_z,const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N){
        d_z[n] = d_x[n] + d_y[n];
    }
}

void timing(const real *h_x, const real *h_y, real *h_z,
    const real *d_x, const real *d_y, real *d_z,
    const int ratio, bool overlap){
        float t_sum = 0;
        float t2_sum = 0;
        for(int repeat = 0; repeat < NUM_REPEATS; ++repeat){
            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            CHECK(cudaEventRecord(start));
            cudaEventQuery(start);

            if(!overlap){
                cpu_sum(h_x, h_y, h_z, N/ratio);
            }
            gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
            if(overlap){
                cpu_sum(h_x, h_y, h_z, N/ratio);
            }

            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float elapsed_time;
            CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
            printf("Time = %g ms.\n", elapsed_time);
            if(repeat > 0){
                t_sum += elapsed_time;
                t2_sum += elapsed_time*elapsed_time;
            }
            CHECK(cudaEventDestroy(start));
            CHECK(cudaEventDestroy(stop));
        }
        const float t_avg = t_sum / (NUM_REPEATS - 1);
        const float t2_avg = sqrt(t2_sum / (NUM_REPEATS - 1) - t_avg * t_avg);
        printf("Average time = %g, error time = %g ms.\n", t_avg,t2_avg);
    }

    int main(void){
        real *h_x = (real *) malloc(M);
        real *h_y = (real *) malloc(M);
        real *h_z = (real *) malloc(M);
        for(int n=0; n<N; n++){
            h_x[n] = 1.23;
            h_y[n] = 2.34;
        }
        real *d_x, *d_y, *d_z;
        CHECK(cudaMalloc(&d_x, M));
        CHECK(cudaMalloc(&d_y, M));
        CHECK(cudaMalloc(&d_z, M));
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

        printf("No overlap:,ratio = 1\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, false);
        printf("Overlap:,ratio = 1\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, true);

        printf("No overlap:,ratio = 2\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
        printf("Overlap:,ratio = 2\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

        printf("No overlap:,ratio = 4\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, false);
        printf("Overlap:,ratio = 4\n");
        timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, true);
        
        free(h_y);
        free(h_z);
        free(h_x);
        CHECK(cudaFree(d_x));
        CHECK(cudaFree(d_y));
        CHECK(cudaFree(d_z));
        return 0;

    }