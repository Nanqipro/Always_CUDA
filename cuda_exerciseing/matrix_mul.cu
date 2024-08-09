#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h> // 确保包含此头文件以使用CUDA函数

// 修正宏定义以包含闭合while循环
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int DSIZE = 8192;
const int block_size = 32;
const float A_val = 3.0f;
const float B_val = 2.0f;

__global__ void matrix_mul(float *C, float *A, float *B, int n) {
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    if((row<n)&&(col<n)){
        float Cvalue = 0;
        for (int k = 0; k < n; ++k) {
            Cvalue += A[row * n + k] *B[k * n + col];
        }
        C[row * n + col] = Cvalue;
    }

}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    // timing
    clock_t t0,t1,t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // start timing
    t0 = clock();
    A = (float*)malloc(DSIZE*DSIZE*sizeof(float));
    B = (float*)malloc(DSIZE*DSIZE*sizeof(float));
    C = (float*)malloc(DSIZE*DSIZE*sizeof(float));
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        A[i] = A_val;
        B[i] = B_val;
        C[i] = 0;
    }
    // Initialize timing
    t1 = clock();
    t1sum = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Initialization time: %f\n", t1sum);

    // Allocate device memory and copy host memory
    cudaMalloc((void**)&d_A, DSIZE*DSIZE*sizeof(float));
    cudaMalloc((void**)&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc((void**)&d_C, DSIZE*DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc");
    cudaMemcpy(d_A, A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");

    // Cuda processing sequence step 1 is completed

    //lanch kernel
    // 定义网格维度大小，根据块大小来确定网格的大小
    dim3 dimGrid(DSIZE/block_size,DSIZE/block_size);
    // 定义块维度大小，设置每个块的大小
    dim3 dimBlock(block_size,block_size);
    // 启动CUDA核函数，进行矩阵乘法操作
    // 参数1：网格维度，参数2：块维度，参数3：目标矩阵设备指针，参数4：矩阵A设备指针，参数5：矩阵B设备指针，参数6：矩阵尺寸
    matrix_mul<<<dimGrid,dimBlock>>>(d_C,d_A,d_B,DSIZE);
    // 检查CUDA核函数调用是否成功
    cudaCheckErrors("kernel launch");

    //Cuda processing sequence step 2 is completed

    // copy result back to host memory
    cudaMemcpy(C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //GPU timing
    t2 = clock();
    t2sum = (double)(t2 - t1) / CLOCKS_PER_SEC;
    printf("GPU time: %f\n", t2sum);

    //Cuda processing sequence step 3 is completed

    // Verify results
    cudaCheckErrors("kernel execution");

    int i;

    for (i = 0; i < DSIZE*DSIZE; i++) {
        if (C[i] != DSIZE*A_val*B_val) {
            printf("Error: result is incorrect.\n");
            break;
        }
    }
    if(i == DSIZE*DSIZE){
        printf("Success: result is correct.\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;

}