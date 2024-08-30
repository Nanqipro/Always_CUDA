#include "error.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>  // 包含CUDA头文件

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

#define CEIL_DIV(x, y) ((x + y - 1) / y)  // 定义CEIL_DIV宏

// inline int CEIL_DIV(int x, int y) {
//     return (x + y - 1) / y;
// }

// 计算矩阵乘法：C = A * B，矩阵A的维度为M*K，矩阵B的维度为K*N
#define M 512
#define K 512
#define N 512

const int NUM_REPEATS = 100000; //重复计算时间的次数

// 初始化矩阵函数
void initial(real *array, int size){
    for(int i = 0; i < size; i++){
        array[i] = (real)(rand() % 10 + 1);
    }
}


__global__ void sgemm_naive(float alpha, real *A, real *B, float beta, real *C){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < M && y < N){
        real Cvalue = 0;
        for(int i = 0; i < K; i++){
            Cvalue += A[x * K + i] * B[i * N + y];
            
        }
        // C = α*(A@B)+β*C
        C[x*N + y] = alpha * Cvalue + beta * C[x*N + y];
       
    }
}

void timing(real *d_A, real *d_B, real *d_C){
    
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32),1);
    dim3 blockDim(32, 32,1);

    float t1_sum = 0.0;
    for(int i = 0; i < NUM_REPEATS; i++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start, 0));
        cudaEventQuery(start);
        sgemm_naive<<<gridDim, blockDim>>>(1.0f, d_A, d_B, 0.0f, d_C);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float t1;
        CHECK(cudaEventElapsedTime(&t1, start, stop));
        printf("%f\n",t1);
        if(i>0){
            t1_sum += t1;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

    }
    printf("Average time: %f\n",t1_sum/(NUM_REPEATS-1));
}

int main(){
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;
    float *A, *B, *C;
    A = (real*)malloc(size_A * sizeof(real));
    B = (real*)malloc(size_B * sizeof(real));
    C = (real*)malloc(size_C * sizeof(real));
    initial(A, size_A);
    initial(B, size_B);

    real *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(real));  
    cudaMalloc((void**)&d_B, size_B * sizeof(real));
    cudaMalloc((void**)&d_C, size_C * sizeof(real));

    cudaMemcpy(d_A, A, size_A * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B * sizeof(real), cudaMemcpyHostToDevice);

    // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32),1);
    // dim3 blockDim(32, 32,1);
    // sgemm_naive<<<gridDim, blockDim>>>(1.0f, d_A, d_B, 0.0f, d_C);
    timing(d_A,d_B,d_C);

    cudaMemcpy(C, d_C, size_C * sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}
