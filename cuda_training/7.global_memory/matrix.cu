#include "error.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real *d_A,real *d_B, int N, int task);
__global__ void copy(const real *A, real *B, int N);
__global__ void transpose1(const real *A,real *B, int N);
__global__ void transpose2(const real *A,real *B, int N);
__global__ void transpose3(const real *A,real *B, int N);


void print_matrix(real *B, int N);

int main(int argc, char **argv) {
    int N = 1 << 10;
    if (argc > 1) {
        N = 1 << atoi(argv[1]);
    }
    int N2 = N * N;
    // int M = sizeof(real) * N2;

    real *h_A = (real *)malloc(N * N * sizeof(real));
    real *h_B = (real *)malloc(N * N * sizeof(real));

    // 初始化向量h_A和h_B的元素
    for (int i = 0; i < N2; i++) {
        h_A[i] = rand() / (real)RAND_MAX; // 为h_A数组的每个元素赋予一个0到1之间的随机数
    }

    for (int i = 0; i < N2; i++) {
        h_B[i] = 0;
    }

     real *d_A, *d_B;
     CHECK(cudaMalloc((void **)&d_A, N * N * sizeof(real)));
     CHECK(cudaMalloc((void **)&d_B, N * N * sizeof(real)));

     CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(real), cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(real), cudaMemcpyHostToDevice));

     printf("\ncopy matrix\n");
     timing(d_A, d_B, N, 1);
     printf("\ntranspose1 matrix\n");
     timing(d_A, d_B, N, 2);
     printf("\ntranspose2 matrix\n");
     timing(d_A, d_B, N, 3);
     printf("\ntranspose3 matrix\n");
     timing(d_A, d_B, N, 4);

     if(N<=10){
        printf("A = \n");
        print_matrix(h_A, N);
        printf("B = \n");
        print_matrix(h_B, N);
     }

     CHECK(cudaFree(d_A));
     CHECK(cudaFree(d_B));
     free(h_A);
     free(h_B);
     return 0;
}

void timing(const real *d_A,real *d_B, int N, int task) {
    int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    int grid_size_y = (N + TILE_DIM - 1) / TILE_DIM;
    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(TILE_DIM, TILE_DIM);

    float t1_sum = 0;
    float t2_sum = 0;

    for (int i = 0; i < NUM_REPEATS; i++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start, 0));
        cudaEventQuery(start);

        switch (task) {
            case 1:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 4:
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("task error\n");
                break;
        }
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float t1;
        CHECK(cudaEventElapsedTime(&t1, start, stop));
        printf("%f ms\n", t1);

        if (i > 0) {//忽略第一次记录的时间
            t1_sum += t1;
            t2_sum += t1*t1;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));


    }
    // 平均用时和标准差计算
    printf("Average time = %f ms\n", t1_sum / (NUM_REPEATS - 1)); // 修正平均值计算
    printf("Standard deviation = %f ms\n", sqrt(t2_sum / (NUM_REPEATS - 1) - (t1_sum / (NUM_REPEATS - 1)) * (t1_sum / (NUM_REPEATS - 1)))); // 修正标准差计算

}

void print_matrix(real *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", B[i * N + j]);
        }
        printf("\n");
    }
}

__global__ void copy(const real *A, real *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        B[i * N + j] = A[i * N + j];
    }
}

__global__ void transpose1(const real *A, real *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        B[i * N + j] = A[j * N + i];
    }
}

__global__ void transpose2(const real *A, real *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        B[j * N + i] = A[i * N + j];
    }
}

__global__ void transpose3(const real *A, real *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        B[j * N + i] = __ldg(&A[i * N + j]);
    }
}

