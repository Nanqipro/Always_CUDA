#include "error.cuh"
#include <stdio.h>

const unsigned FULL_MASK = 0xffffffff;  
const unsigned BLOCK_SIZE = 16;
const unsigned WIDTH = 8;

void __global__ test_warp_primitives(void);

int main (int argc, char **argv){
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize());
    return 0;
}

__global__ void test_warp_primitives(void){
    int tid = threadIdx.x;
    int lane_id = tid % WIDTH;

    //这部分代码输出每个线程的ID，并将其格式化输出。如果tid为0，则输出列头。
    if(tid == 0){
        printf("threadIdx.x: ");
    }
    printf("%2d ", tid);
    if(tid == 0){
        printf("\n");
    }
    //这部分代码输出每个线程的lane_id，并将其格式化输出。如果tid为0，则输出列头。
    if(tid == 0){
        printf("lane_id: ");
    }
    printf("%2d ", lane_id);
    if(tid == 0){
        printf("\n");
    }

    unsigned mask1 = __ballot_sync(FULL_MASK, tid >0);
    unsigned mask2 = __ballot_sync(FULL_MASK, tid == 0);
    if(tid == 0){
        printf("Full mask: %x\n", FULL_MASK);
    }
    if(tid == 1){
        printf("mask1: %x\n", mask1);
    }
    if(tid == 0){
        printf("mask2: %x\n", mask2);
    }

    int result = __all_sync(FULL_MASK, tid);
    if(tid == 0){
        printf("__all_sync(FULL_MASK, tid): %d\n", result);
    }

    result = __all_sync(mask1, tid);
    if(tid == 1){
        printf("__all_sync(mask1, tid): %d\n", result);
    }

    result = __any_sync(FULL_MASK, tid);
    if(tid == 0){
        printf("__any_sync(FULL_MASK, tid): %d\n", result);
    }

    result = __any_sync(mask2, tid);
    if(tid == 0){
        printf("__any_sync(mask2, tid): %d\n", result);
    }
    
    int val = __shfl_sync(FULL_MASK, tid, 2,WIDTH);
    if(tid == 0){
        printf("__shfl:  ");
    }
    printf("%2d ", val);
    if(tid == 0){
        printf("\n");
    }

    val = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if(tid == 0){
        printf("__shfl_up:  ");
    }
    printf("%2d ", val);
    if(tid == 0){
        printf("\n");
    }

    val = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if(tid == 0){
        printf("__shfl_down:  ");
    }
    printf("%2d ", val);
    if(tid == 0){
        printf("\n");
    }

    val = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if(tid == 0){
        printf("__shfl_xor:  ");
    }
    printf("%2d ", val);
    if(tid == 0){
        printf("\n");
    }
}