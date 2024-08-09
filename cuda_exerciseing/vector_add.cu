#include <stdio.h>
#include <stdlib.h>

// 错误检查宏
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

const int DSIZE = 4096;
const int block_size = 256;

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_a, *d_b, *d_c;

    h_A = (float*)malloc(DSIZE * sizeof(float));
    h_B = (float*)malloc(DSIZE * sizeof(float));
    h_C = (float*)malloc(DSIZE * sizeof(float));

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
        h_C[i] = 0;
    }

    cudaMalloc((void**)&d_a, DSIZE * sizeof(float));
    cudaMalloc((void**)&d_b, DSIZE * sizeof(float));
    cudaMalloc((void**)&d_c, DSIZE * sizeof(float));

    cudaCheckErrors("cudaMalloc failed");

    cudaMemcpy(d_a, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy failed");

    vector_add<<<DSIZE / block_size, block_size>>>(d_a, d_b, d_c, DSIZE);

    cudaCheckErrors("kernel launch failed");

    cudaMemcpy(h_C, d_c, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckErrors("cudaMemcpy failed");

    for (int i = 0; i < DSIZE; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
