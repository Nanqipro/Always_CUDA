#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 30;
const int N = N1 * MAX_NUM_STREAMS;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N1 - 1) / block_size + 1;
cudaStream_t streams[MAX_NUM_STREAMS];


void __global__ add(const real *d_x, const real *d_y, real *d_z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N1)
    {   // 循环100000次的目的是模拟了一个高计算负载的场景
        for (int i = 0; i < 100000; ++i)
        {
            d_z[n] = d_x[n] + d_y[n];
        }
    }
}

void timing(const real *d_x,const real *d_y, real *d_z, const int N){
    float t_sum = 0;
    float t2_sum = 0;
    for(int i = 0; i < NUM_REPEATS; ++i){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start, 0));
        cudaEventQuery(start);

        for(int j = 0; j < N; ++j){
            int offset = j * N1;
            add<<<grid_size, block_size,0,streams[j]>>>(d_x+offset, d_y+offset, d_z+offset);
        }
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        if(i>0){
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    const float t_avg = t_sum / (NUM_REPEATS - 1);
    const float t2_avg = sqrt(t2_sum / (NUM_REPEATS - 1) - t_avg * t_avg);
    printf("Average time = %f ms\n", t_avg);
    printf("Standard deviation = %f ms\n", t2_avg);
}

int main(void){
    real *h_x = (real *)malloc(M);
    real *h_y = (real *)malloc(M);
    // real *h_z = (real *)malloc(M);

    for(int i = 0; i < N; ++i){
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
    for(int i = 0; i < MAX_NUM_STREAMS; ++i){
        CHECK(cudaStreamCreate(&streams[i]));
    }
    for(int i = 1; i <= MAX_NUM_STREAMS; ++i){
        timing(d_x, d_y, d_z, i);
    }
    for(int i = 0; i < MAX_NUM_STREAMS; ++i){
        CHECK(cudaStreamDestroy(streams[i]));
    }
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    free(h_x);
    free(h_y);
    return 0;

}