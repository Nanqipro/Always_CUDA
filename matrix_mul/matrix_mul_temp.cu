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

// GPU V2 --使用共享内存和分段方法
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

// GPU V3 --一个线程处理多个元素
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrix_mul_03(float *dA, float *dB, float *dC, int m, int k, int n){
    __shared__ float SA[BM*BK];
    __shared__ float SB[BK*BN];

    int indA = TM*(threadIdx.x+blockIdx.x*blockDim.x);
    int indB = TN*(threadIdx.y+blockIdx.y*blockDim.y);

    // 分成多少段
    int width = (k+BK-1)/BK;
     
    // 初始化
    float tmp[TM*TN] = {0.0f};

    for(int ph=0;ph<width;ph++){
        // 加载矩阵到共享内存
        for(int index_q = 0 ; index_q<TM ; index_q++){
            for(index_k = 0 ; index_k<BK ; index_k++){
                if(indA+index_q<M && index_k+ph*BK<K){
                    SA[(threadIdx.x*TM+index_q)*BK+index_k] = dA[(indA+index_q)*K+index_k+ph*BK];
                }else{
                    SA[(threadIdx.x*TM+index_q)*BK+index_k] = 0.0f;
                }
            }
        }
        __syncthreads();
        for(int index_p= 0;index_p<TN;index_p++){
            for(index_k = 0;index_k<BK;index_k++){
                if(indB+index_p<N && index_k+ph*BK<K){
                    SB[index_k*BN+ threadIdx.y*TN + index_p] = dB[(index_k+ph*BK)*N + indB+index_p];
                }else{
                    SB[index_k*BN+ threadIdx.y*TN + index_p] = 0.0f;
                }
            }
        }
        __syncthreads();
        // 计算
        for(int index_q=0;index_q<TM;index_q++){
            for(int index_p=0;index_p<TN;index_p++){
                for(int index_k=0;index_k<BK;index_k++){
                    tmp[index_q*TN + index_p] += SA[(threadIdx.x*TM+index_q)*BK+index_k]*SB[index_k*BN+threadIdx.y*TN+index_p];
                }
            }
        }
        __syncthreads();

    }

    for(int index_q=0;index_q<TM;index_q++){
        for(int index_p=0;index_p<TN;index_p++){
            if(indA+index_q<M && indB+index_p<N){
                dC[(indA+index_q)*N+indB+index_p] = tmp[index_q*TN+index_p];
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
    // matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        // matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
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