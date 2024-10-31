#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>

const int TM = 4;
const int TN = 4;
const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 32;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;


# 矩阵乘法及其优化实现

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

// cpu 上矩阵乘法
void matrix_mul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for(int i= 0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum = 0;
            for(int l=0;l<k;l++){
                sum += A[i*k+l]*B[l*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}
// GPU V1
__global__ void matrix_mul_01(float *A, float *B, float *C, int m, int k, int n){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if(row<m && col<n){
        float sum = 0;
        for(int l=0;l<k;l++){
            sum += A[row*k+l]*B[l*n+col];
        }
        C[row*n+col] =sum;
    }
}

// GPU V2
template <int BLOCK_DIM>
__global__ void matrix_mul_02(float *dA, float *dB, float *dC, int m, int k, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0.0f;

    __shared__ float s_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_B[BLOCK_DIM][BLOCK_DIM];

    int width = (k+BLOCK_DIM-1)/BLOCK_DIM;

    for(int i=0;i<width;i++){
        if(row<m && i*BLOCK_DIM+threadIdx.y<k){
            s_A[threadIdx.x][threadIdx.y] = dA[row*k+i*BLOCK_DIM+threadIdx.y];
        }else{
            s_A[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if(col<n && i*BLOCK_DIM+threadIdx.x<k){
            s_B[threadIdx.x][threadIdx.y] = dB[(i*BLOCK_DIM+threadIdx.x)*n+col];
        }else{
            s_B[threadIdx.x][threadIdx.y] = 0.0f;
        }
    }
    __syncthreads();
    for(int j=0;j<BLOCK_DIM;j++){
        tmp += s_A[threadIdx.x][j]*s_B[j][threadIdx.y];
    }
    __syncthreads();

    if(row<m && col<n){
        dC[row*n+col] = tmp;
    }
}

// GPU V3
template <int BLOCK_DIM>
__global__ void matrix_mul_03(float *dA, float *dB, float *dC, int m, int k, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

}
