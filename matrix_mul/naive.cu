// #include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>  // 包含CUDA头文件
// #define CEIL_DIV(x, y) ((x + y - 1) / y)  // 定义CEIL_DIV宏

inline int CEIL_DIV(int x, int y) {
    return (x + y - 1) / y;
}

// 计算矩阵乘法：C = A * B，矩阵A的维度为M*K，矩阵B的维度为K*N
#define M 512
#define K 512
#define N 512

// const int NUM_REPEATS = 100; //重复计算时间的次数

// 初始化矩阵函数
void initial(float *array, int size){
    for(int i = 0; i < size; i++){
        array[i] = (float)(rand() % 10 + 1);
    }
}


__global__ void sgemm_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < M && y < N){
        float Cvalue = 0;
        for(int i = 0; i < K; i++){
            Cvalue += A[x * K + i] * B[i * N + y];
            
        }
        // C = α*(A@B)+β*C
        C[x*N + y] = alpha * Cvalue + beta * C[x*N + y];
       
    }
}

int main(){
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;
    float *A, *B, *C;
    A = (float*)malloc(size_A * sizeof(float));
    B = (float*)malloc(size_B * sizeof(float));
    C = (float*)malloc(size_C * sizeof(float));
    initial(A, size_A);
    initial(B, size_B);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(float));  
    cudaMalloc((void**)&d_B, size_B * sizeof(float));
    cudaMalloc((void**)&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32),1);
    dim3 blockDim(32, 32,1);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaMemcpy(C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}
