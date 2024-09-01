// #include <cuda.h>
// #include <sys/time.h>
// #include <stdio.h>
// const int TM = 4;
// const int TN = 4;
// const int BLOCK_DIM_x = 32;
// const int BLOCK_DIM_y = 32;
// const int BM = TM * BLOCK_DIM_x;
// const int BN = TN * BLOCK_DIM_y;
// const int BK = 8;
// double
// get_walltime()
// {
//     struct timeval tp;
//     gettimeofday(&tp, NULL);
//     return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
// }
// // 不用GPU的矩阵乘法
// void matrix_mul_naive(float *A, float *B, float *C, int m, int k, int n){
//     for(int i = 0; i < m; i++){
//         for(int j = 0; j < n; j++){
//             float sum = 0;
//             for(int l = 0; l < k; l++){
//                 sum += A[i * k + l] * B[l * n + j];
//             }
//             C[i * n + j] = sum;
//         }
//     }
// }
// // 比较两种算法的差异，用于判断是否正确
// void compare(float *hostC, float *serialC, int M, int N)
// {
//     float error = 0;
//     float err = 0;
//     bool tmp = true;
//     for (int i = 0; i < M * N; i++)
//     {
//         // error = fmax(error, fabs(hostC[i] - serialC[i]));
//         err = fabs(hostC[i] - serialC[i]);
//         if (err > 1e-5)
//         {
//             tmp = false;
//             printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
//             // break;
//         }
//     }
//     if (tmp)
//     {
//         printf("GPU output all right\n");
//     }
// }


// // 用GPU进行矩阵乘法v3--瓦片化（tiling）方法
// template<int BM,int BN,int BK,int TM,int TN>
// __global__ void matrix_mul_03(float *A, float *B, float *C, int m, int k, int n){
//     __shared__ float s_A[BM*BK];
//     __shared__ float s_B[BK*BN];
//     //indA 和 indB 分别表示矩阵A和B中当前线程块开始处理的位置。
//     int indA = TM * (threadIdx.x+blockIdx.x*blockDim.x);
//     int indB = TN * (threadIdx.y+blockIdx.y*blockDim.y);
//     int width = (k+BK-1)/BK;//处理的总瓦片数
//     float tmp[TM*TN] = {0.0f};//初始化

//     for(int i=0;i<width;i++){
//         //把A加载到共享内存
//         for(int index_q = 0;index_q < TM ;index_q++){
//             for(int index_k = 0;index_k < BK;index_k++){
//                 if(index_q+indA < m && index_k+i*BK < k) {
//                     s_A[(threadIdx.x* TM + index_q)*BK + index_k] = A[(index_q+indA)*k + index_k+i*BK];
//                 }
//                 else{
//                     s_A[(threadIdx.x* TM + index_q)*BK + index_k] = 0.0f;
//                 } 
//             }
//         }
//         __syncthreads();
//         //把B加载到共享内存
//         for(int index_v = 0;index_v < TN ;index_v++){
//             for(int index_k = 0;index_k < BK;index_k++){
//                 if(index_v+indB < n && index_k+i*BK < k) {
//                     s_B[(threadIdx.y* TN + index_v)*BK + index_k] = B[(index_v+indB)*k + index_k+i*BK];
//                 }
//                 else{
//                     s_B[(threadIdx.y* TN + index_v)*BK + index_k] = 0.0f;
//                 }
//             }
//         }
//         __syncthreads();
//         //瓦片的计算
//         for(int i = 0;i < TM;i++){
//             for(int j = 0;j < TN;j++){
//                 for(int k = 0;k < BK;k++){
//                     tmp[i*TN+j] += s_A[(threadIdx.x*TM+i)*BK+k] * s_B[k * BN + threadIdx.y * TN + j];
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     //存储结果
//     for(int i = 0;i < TM;i++){
//         for(int j = 0;j < TN;j++){
//             if(indA+i < m && indB+j < n){
//                 C[(indA+i)*n + indB+j] = tmp[i*TN+j];
//             }
//         }
//     }
// }



// void hostMatrix(float *A, float *B, float *C, int m, int k, int n){
//     double start = get_walltime();
//     double elapsed = 0;

//     float *d_A, *d_B, *d_C;
//     cudaMalloc((void **)&d_A, m * k * sizeof(float));
//     cudaMalloc((void **)&d_B, k * n * sizeof(float));
//     cudaMalloc((void **)&d_C, m * n * sizeof(float));

//     cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);


//     int num_blocks_x = (m + BM - 1) / BM;
//     int num_blocks_y = (n + BN - 1) / BN;

//     dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y,1);
//     dim3 grid_dim(num_blocks_x, num_blocks_y,1);

//     int NUM_REPEATS = 20;
//     // matrix_mul_01<<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//     // matrix_mul_02<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//     matrix_mul_03<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//     cudaEvent_t start_event, stop_event;
//     float kernel_time = 0;
//     cudaEventCreate(&start_event);
//     cudaEventCreate(&stop_event);
//     cudaEventRecord(start_event,0);
    
//     for(int i = 0; i < NUM_REPEATS; i++){
//         // matrix_mul_01<<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//         // matrix_mul_02<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//         matrix_mul_03<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(d_A, d_B, d_C, m, k, n);
//     }
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//         // Possibly: exit(-1) if program cannot continue....
//     }

//     cudaEventRecord(stop_event,0);
//     cudaEventSynchronize(stop_event);
//     cudaEventElapsedTime(&kernel_time, start_event, stop_event);

//     cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     cudaEventDestroy(start_event);
//     cudaEventDestroy(stop_event);
//     elapsed = get_walltime() - start;
//     printf("M-K-N: %d-%d-%d\n", m, k, n);
//     printf("GPU use time: %.4f second\n", elapsed);
//     printf("kernel time: %.4f second, %.4f ms\n", kernel_time / (NUM_REPEATS * 1000.), kernel_time / NUM_REPEATS);
//     printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
//     printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
// }

// int main(int argc, char **argv){
//     float *A, *B, *C, *serialC;
//     int M = 1024;
//     int K = 1024;
//     int N = 1024;
//     A = (float *)malloc(M * K * sizeof(float));
//     B = (float *)malloc(K * N * sizeof(float));
//     C = (float *)malloc(M * N * sizeof(float));
//     serialC = (float *)malloc(M * N * sizeof(float));
//     //初始化
//     for (int i = 0; i < M * K; i++)
//     {
//         A[i] = i%3;
//     }
//     for (int i = 0; i < K * N; i++)
//     {
//         B[i] = i%3;
//     }
//     //用GPU进行矩阵乘法
//     hostMatrix(A, B, C, M, K, N);

//     double start = get_walltime();
//     double elapsed = 0;
//     //不用GPU进行矩阵乘法
//     matrix_mul_naive(A, B, serialC, M, K, N);

//     elapsed = get_walltime() - start;
//     //比较计算是否正确
//     printf("CPU use time: %.4f second\n", elapsed);
//     compare(C, serialC, M, N);
//     free(A);
//     free(B);
//     free(C);
//     free(serialC);
//     return 0;

// }



#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
const int TM = 4;
const int TN = 4;
const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 32;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;
double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
void compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    bool tmp = true;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
        if (error > 1e-5)
        {
            tmp = false;
            printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }
    if (tmp)
    {
        printf("GPU output all right\n");
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};

    for (int ph = 0; ph < width; ph++)
    {

        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_k = 0; index_k < BK; index_k++)
            {
                if (indA + index_q < M && index_k + ph * BK < K)
                {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = dA[(indA + index_q) * K + index_k + ph * BK];
                }
                else
                {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = 0.0f;
                }
            }
        }
        __syncthreads();
        for (int index_v = 0; index_v < TN; index_v++)
        {
            for (int index_k = 0; index_k < BK; index_k++)
            {

                if (indB + index_v < N && index_k + ph * BK < K)
                {

                    SB[index_k * BN + threadIdx.y * TN + index_v] = dB[(index_k + ph * BK) * N + indB + index_v];
                }
                else
                {
                    SB[index_k * BN + threadIdx.y * TN + index_v] = 0.0f;
                }
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[(threadIdx.x * TM + index_q) * BK + index_k] * SB[index_k * BN + threadIdx.y * TN + index_v];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q * TN + index_v];
            }
        }
    }
}


void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    hostMatrix(hostA, hostB, hostC, M, K, N);
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);
    compare(hostC, serialC, M, N);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}