#include "error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;
// using namespace std;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

real reduce(const real *d_x, const int method);
void timing(const real *d_x, const int method);
__global__ void reduce_syncwarp(const real *d_x, real *d_y, const int N);
__global__ void reduce_shfl(const real *d_x, real *d_y, const int N);
__global__ void reduce_cp(const real *d_x, real *d_y, const int N);

real reduce(const real *d_x, const int method)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method)
    {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 2:
            reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        default:
            printf("Wrong method.\n");
            exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x, const int method)
{
    real sum = 0;
    
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method); 

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

__global__ void reduce_syncwarp(const real *d_x, real *d_y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid*blockDim.x+tid;

    extern __shared__ real s_y[];
    s_y[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();

    for(int i=blockDim.x/2; i>=32; i/=2){
        if(tid<i){
            s_y[tid] += s_y[tid+i];
        }
        __syncthreads();
    }
    for(int i = 16; i>=1; i/=2){
        if(tid<i){
            s_y[tid] += s_y[tid+i];
        }
        __syncwarp(FULL_MASK);
    }
    if(tid==0){
        atomicAdd(d_y, s_y[0]);
    }

}

__global__ void reduce_shfl(const real *d_x, real *d_y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x; 
    const int n = bid*blockDim.x+tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();
    for(int i=blockDim.x/2; i>=32; i/=2){
        if(tid<i){
            s_y[tid] += s_y[tid+i];
        }
        __syncthreads();
    }
    real sum = s_y[tid];
    for(int offset = 16; offset>0; offset/=2){
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if(tid==0){
        atomicAdd(d_y, sum);
    }
}

__global__ void reduce_cp(const real *d_x, real *d_y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x; 
    const int n = bid*blockDim.x+tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();
    for(int i=blockDim.x/2; i>=32;i/=2){
        if(tid<i){
            s_y[tid] += s_y[tid+i];
        }
        __syncthreads();
    }
    real sum = s_y[tid];
    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block());
    for(int offset = tile.size(); offset>0; offset/=2){
        sum += tile.shfl_down(sum, offset);
    }
    
    if(tid==0){
        atomicAdd(d_y, sum);
    }
}

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing syncwarp:\n");
    timing(d_x, 0);
    printf("\nusing shfl:\n");
    timing(d_x, 1);
    printf("\nusing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}






