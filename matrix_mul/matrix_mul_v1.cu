#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

// 不用GPU的矩阵乘法
void matrix_mul_naive(float *A, float *B, float *C, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float sum = 0;
            for(int l = 0; l < k; l++){
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 比较两种算法的差异，用于判断是否正确
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}

// 用GPU进行矩阵乘法v1

__global__ void matrix_mul_01(float *A, float *B, float *C, int m, int k, int n){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if(row < m && col < n){
        float sum = 0;
        for(int l = 0; l < k; l++){
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] =sum;
    }
}

// 用GPU进行矩阵乘法v2
template <int BLOCK_DIM> //模板函数
__global__ void matrix_mul_02(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0.0f;
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM][BLOCK_DIM];
    int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    for (int ph = 0; ph < width; ph++)
    {
        if (row < M && threadIdx.y + ph * BLOCK_DIM < K)
        {
            SA[threadIdx.x][threadIdx.y] = dA[row * K + threadIdx.y + ph * BLOCK_DIM];
        }
        else
        {
            SA[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if (threadIdx.x + ph * BLOCK_DIM < K && col < N)
        {
            SB[threadIdx.x][threadIdx.y] = dB[(threadIdx.x + ph * BLOCK_DIM) * N + col];
        }
        else
        {
            SB[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int s = 0; s < BLOCK_DIM; s++)
        {
            tmp += SA[threadIdx.x][s] * SB[s][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        dC[row * N + col] = tmp;
    }
}

void hostMatrix(float *A, float *B, float *C, int m, int k, int n){
    double start = get_walltime();
    double elapsed = 0;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_blocks_x = (m + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (n + BLOCK_DIM_y - 1) / BLOCK_DIM_y;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y,1);
    dim3 grid_dim(num_blocks_x, num_blocks_y,1);

    int NUM_REPEATS = 100;
    matrix_mul_01<<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
    // matrix_mul_02<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
    cudaEvent_t start_event, stop_event;
    float kernel_time = 0;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event,0);
    
    for(int i = 0; i < NUM_REPEATS; i++){
        matrix_mul_01<<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
        // matrix_mul_02<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);

    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    elapsed = get_walltime() - start;
    printf("M-K-N: %d-%d-%d\n", m, k, n);
    printf("GPU use time: %.4f second\n", elapsed);
    printf("kernel time: %.4f second, %.4f ms\n", kernel_time / (NUM_REPEATS * 1000.), kernel_time / NUM_REPEATS);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main(int argc, char **argv){
    float *A, *B, *C, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    //初始化
    for (int i = 0; i < M * K; i++)
    {
        A[i] = i%3;
    }
    for (int i = 0; i < K * N; i++)
    {
        B[i] = i%3;
    }
    //用GPU进行矩阵乘法
    hostMatrix(A, B, C, M, K, N);

    double start = get_walltime();
    double elapsed = 0;
    //不用GPU进行矩阵乘法
    matrix_mul_naive(A, B, serialC, M, K, N);

    elapsed = get_walltime() - start;
    //比较计算是否正确
    float error = compare(C, serialC, M, N);
    printf("CPU use time: %.4f second\n", elapsed);
    printf("error: %.4f\n", error);
    free(A);
    free(B);
    free(C);
    free(serialC);
    return 0;

}


