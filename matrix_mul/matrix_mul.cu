#include <stdio.h>
#include "error.cuh"
#include <cuda_runtime.h>
// 矩阵乘法的朴素实现
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

// 计算矩阵乘法：C = A * B，矩阵A的维度为M*K，矩阵B的维度为K*N

#define M 512
#define K 512
#define N 512

const int NUM_REPEATS = 10;

void initial(real *array, int size){
    for(int i = 0; i < size; i++){
        array[i] = (real)(rand() % 10 + 1);
    }
}


void print_matrix(real *C, int m,int n){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }
}

__global__ void matrix_mul_01(real *A, real *B,real *C,int m, int k, int n){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if(ix < m && iy < n){
        real Cvalue = 0;
        for(int kk = 0; kk < k; kk++){
            Cvalue += A[ix * k + kk] * B[kk * n + iy]; //行优先矩阵
        }
        C[ix * n + iy] = Cvalue;
    }
}


void timing(real *d_A, real *d_B, real *d_C, int m, int k, int n){
    int dimx = 2;
    int dimy = 2;

    dim3 dimBlock(dimx,dimy);
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);

    float t1_sum = 0.0;
    for(int i = 0; i < NUM_REPEATS; i++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start, 0));
        cudaEventQuery(start);
        matrix_mul_01<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,M,K,N);
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

int main(int argc, char **argv){
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;
    real *A, *B, *C;
    A = (real*)malloc(size_A * sizeof(real));
    B = (real*)malloc(size_B * sizeof(real));
    C = (real*)malloc(size_C * sizeof(real));

    initial(A, size_A);
    initial(B, size_B);

    // print_matrix(A,M,K);
    // print_matrix(B,K,N);
    
    real *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(real));  
    cudaMalloc((void**)&d_B, size_B * sizeof(real));
    cudaMalloc((void**)&d_C, size_C * sizeof(real));

    cudaMemcpy(d_A, A, size_A * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B * sizeof(real), cudaMemcpyHostToDevice);

    // int dimx = 2;
    // int dimy = 2;

    // dim3 dimBlock(dimx,dimy);
    // dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
    // matrix_mul_01<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,M,K,N);

    // print_matrix(C,M,N);

    timing(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(C, d_C, size_C * sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}

// #include <stdio.h>
// #include "error.cuh"
// #include <cuda_runtime.h>

// // 矩阵乘法的朴素实现
// // 计算矩阵乘法：C = A * B，矩阵A的维度为M*K，矩阵B的维度为K*N

// #define M 512
// #define K 512
// #define N 512

// const int NUM_REPEATS = 10;

// void initial(float *array, int size){
//     for(int i = 0; i < size; i++){
//         array[i] = (float)(rand() % 10 + 1);
//     }
// }

// void print_matrix(float *C,int m,int n){
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%f ", C[i * N + j]);
//         }
//         printf("\n");
//     }
// }

// __global__ void matrix_mul_01(float *A, float *B, float *C, int m, int k, int n){
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iy = blockIdx.y * blockDim.y + threadIdx.y;
//     if(ix < m && iy < n){
//         float Cvalue = 0;
//         for(int kk = 0; kk < k; kk++){
//             Cvalue += A[ix * k + kk] * B[kk * n + iy]; //行优先矩阵
//         }
//         C[ix * n + iy] = Cvalue;
//     }
// }

// void timing(float *d_A, float *d_B, float *d_C, int m, int k, int n){
//     int dimx = 2;
//     int dimy = 2;

//     dim3 dimBlock(dimx,dimy);
//     dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);

//     float t1_sum = 0.0;
//     for(int i = 0; i < NUM_REPEATS; i++){
//         cudaEvent_t start, stop;
//         CHECK(cudaEventCreate(&start));
//         CHECK(cudaEventCreate(&stop));
//         CHECK(cudaEventRecord(start, 0));
//         cudaEventQuery(start);
//         matrix_mul_01<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,M,K,N);
//         CHECK(cudaEventRecord(stop, 0));
//         CHECK(cudaEventSynchronize(stop));
//         float t1;
//         CHECK(cudaEventElapsedTime(&t1, start, stop));
//         printf("%f\n",t1);
//         if(i>0){
//             t1_sum += t1;
//         }
//         CHECK(cudaEventDestroy(start));
//         CHECK(cudaEventDestroy(stop));
//     }
//     printf("Average time: %f\n",t1_sum/(NUM_REPEATS-1));
// }

// int main(int argc, char **argv){
//     int size_A = M * K;
//     int size_B = K * N;
//     int size_C = M * N;
//     float *A, *B, *C;
//     A = (float*)malloc(size_A * sizeof(float));
//     B = (float*)malloc(size_B * sizeof(float));
//     C = (float*)malloc(size_C * sizeof(float));

//     initial(A, size_A);
//     initial(B, size_B);

//     print_matrix(A,M,K);
//     print_matrix(B,K,N);
    
//     float *d_A, *d_B, *d_C;
//     cudaMalloc((void**)&d_A, size_A * sizeof(float));  
//     cudaMalloc((void**)&d_B, size_B * sizeof(float));
//     cudaMalloc((void**)&d_C, size_C * sizeof(float));

//     cudaMemcpy(d_A, A, size_A * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, size_B * sizeof(float), cudaMemcpyHostToDevice);

//     int dimx = 2;
//     int dimy = 2;

//     dim3 dimBlock(dimx,dimy);
//     dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
//     matrix_mul_01<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,M,K,N);

//     print_matrix(C,M,N);

//     timing(d_A, d_B, d_C, M, K, N);

//     cudaMemcpy(C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     free(A);
//     free(B);
//     free(C);
//     return 0;
// }
