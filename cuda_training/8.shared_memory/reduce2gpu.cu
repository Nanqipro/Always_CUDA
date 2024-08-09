#include "error.cuh"
#include <stdio.h>
#include <cuda_runtime.h> // 添加cuda_runtime.h头文件以包含CUDA API

//实现数组归约计算

#ifdef  USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 1000000000;
const size_t M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x,real *d_x, const int method);
void __global__ reduce_global(real *d_x, real *d_y);
void __global__ reduce_shared(real *d_x, real *d_y);
void __global__ reduce_dynamic(real *d_x, real *d_y);
void __global__ reduce_atomic(real *d_x, real *d_y, const int N);
real reduce(real *d_x, const int method);
real reduce_a(const real *d_x);

int main(void){
    real *h_x = (real *)malloc(M);
    for(int i = 0; i < N; i++){
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, M));
    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing shared memory only:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);
    CHECK(cudaFree(d_x));
    free(h_x);
    return 0;
}

real reduce(real *d_x, const int method){
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // int block_size = BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc((void **)&d_y, ymem));
    real *h_y = (real *)malloc(ymem);
    switch(method){
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size, BLOCK_SIZE,smem>>>(d_x, d_y);
            break;
        default:
            printf("Invalid method\n");
            exit(1);
            break;
    }
    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
    real result = 0.0;
    for(int i = 0; i < grid_size; i++){
        result += h_y[i];
    }
    CHECK(cudaFree(d_y));
    free(h_y);
    return result;
}

real reduce_a(real *d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce_atomic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}


void timing(real *h_x, real *d_x, const int method){
    real sum1 = 0;
    real sum2 = 0;
    for(int i = 0; i < NUM_REPEATS; i++){
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start, 0));
        cudaEventQuery(start);
        sum1 = reduce(d_x, method);//计算
        sum2 = reduce_a(d_x);//原子计算
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("Time = %f\n", elapsedTime);
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        printf("Sum1 = %f\n", sum1);
        printf("Sum2 = %f\n", sum2);
    }

}

__global__ void reduce_atomic(real *d_x, real *d_y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid*blockDim.x+tid;
    extern __shared__ real sdata[];
    sdata[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(d_y, sdata[0]);
    }
}

__global__ void reduce_global(real *d_x, real *d_y){
    const int tid = threadIdx.x;
    real *x = d_x + blockIdx.x * blockDim.x;
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_y[blockIdx.x] = x[0];
    }
}

__global__ void reduce_shared(real *d_x, real *d_y){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid*blockDim.x+tid;
    __shared__ real sdata[BLOCK_SIZE];
    sdata[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();

   for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
       if(tid < offset){
           sdata[tid] += sdata[tid + offset];
       }
       __syncthreads();
   }
   if(tid == 0){
       d_y[bid] = sdata[0];
   }
}

__global__ void reduce_dynamic(real *d_x, real *d_y){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid*blockDim.x+tid;
    extern __shared__ real sdata[];
    sdata[tid] = (n<N)? d_x[n] : 0.0;
    __syncthreads();
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_y[bid] = sdata[0];
    }
}
