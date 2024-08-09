#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;// 记录线程块的索引值
    const int tid = threadIdx.x;// 记录线程的索引值

    const int id = threadIdx.x + blockIdx.x * blockDim.x;  // 记录整个网格中的索引值
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}


int main(void)
{
    printf("Hello World from CPU!\n");
    hello_from_gpu<<<2, 2>>>(); // <<<grid_size,block_size>>>
    cudaDeviceSynchronize();

    return 0;
}