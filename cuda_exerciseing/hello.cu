# include <stdio.h>

 __global__ void hello(void)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello, World from block: %d ,thread: %d\n",bid,tid);
}

int main(void)
{
    hello<<<2,2>>>();
    cudaDeviceSynchronize();
    return 0;
}