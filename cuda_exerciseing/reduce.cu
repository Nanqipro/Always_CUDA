#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>


// 向量规约算子


double get_walltime(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)t.tv_sec + (double)t.tv_usec*1e-6;

}

// cpu reduce
float reduce_cpu(float *data, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}

// gpu reduce
template <int BLOCK_DIM>
__global__ void addKernel (float *dA, int n, float *dSum,int way)
// dA 输入向量
// n 向量长度
// dSum 输出向量
// 采用哪种方式
{
    __shared__ float temp[BLOCK_DIM];
    float tmp = 0.0f;
    
    // 每个线程对其负责的数据块进行累加操作
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        tmp += dA[i];
    }
    temp[threadIdx.x] = tmp;
    __syncthreads();
    if(way == 0){// 交叉规约
        for (int i = 1; i < BLOCK_DIM; i *= 2)
        {
            if (threadIdx.x % (2 * i) == 0 && threadIdx.x + i < BLOCK_DIM)
            {
                temp[threadIdx.x] += temp[threadIdx.x + i];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            dSum[0] = temp[0];
        }

    }else if(way == 1){// 交错规约
        for (int i = BLOCK_DIM / 2; i > 0; i >>= 1)
        {
            if (threadIdx.x < i)    
            {
                temp[threadIdx.x] += temp[threadIdx.x + i];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            dSum[0] = temp[0];
        }
    }else if(way == 2){// shuffle warp规约
        __shared__ float val[32];
        float data = temp[threadIdx.x];
        data += __shfl_down_sync(0xffffffff,data,16);
        data += __shfl_down_sync(0xffffffff,data,8);
        data += __shfl_down_sync(0xffffffff,data,4);
        data += __shfl_down_sync(0xffffffff,data,2);
        data += __shfl_down_sync(0xffffffff,data,1);
        if(threadIdx.x%32 == 0){
            val[threadIdx.x/32] == data;
        }
        __syncthreads();
        if(threadIdx.x <32){
            data = val[threadIdx.x];
            data += __shfl_down_sync(0xffffffff,data,16);
            data += __shfl_down_sync(0xffffffff,data,8);
            data += __shfl_down_sync(0xffffffff,data,4);
            data += __shfl_down_sync(0xffffffff,data,2);
            data += __shfl_down_sync(0xffffffff,data,1);
        }
        __syncthreads();
        if(threadIdx.x == 0){
            dSum[0] = data;
        }
    }
    
}

int main(){
    float *hostA;
    int n = 102400;
    int way = 2;
    int repeat = 100;
    hostA = (float *)malloc(n * sizeof(float));

    //初始化数组
    for (int i = 0; i < n; i++)
    {
        hostA[i] = (i % 10) * 1e-1;
    }

    float hostMax;
    double st, ela;
    st = get_walltime();

    float *dA, *globalMax;

    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&globalMax, sizeof(float));
    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    for (int i = 0; i < repeat; i++)
    {
        addKernel<1024><<<grid_dim, block_dim>>>(dA, n, globalMax, way);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(&hostMax, globalMax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(globalMax);
    ela = 1000 * (get_walltime() - st);
    printf("n = %d: strategy:%d, GPU use time:%.4f ms, kernel time:%.4f ms\n", n, way, ela, ker_time / repeat);
    printf("CPU sum:%.2f, GPU sum:%.2f\n", reduce_cpu(hostA, n), hostMax);
    free(hostA);

    return 0;
}




// #include <cuda.h>
// #include <sys/time.h>
// #include <stdio.h>

// double get_walltime()
// {
//     struct timeval tp;
//     gettimeofday(&tp, NULL);
//     return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
// }
// float addCpu(float *hostA, int n)
// {
//     float tmp = 0.0f; // 表示C++中的负无穷
//     for (int i = 0; i < n; i++)
//     {
//         tmp += hostA[i];
//     }
//     return tmp;
// }
// template <int BLOCK_DIM>
// __global__ void addKernel(float *dA, int n, float *globalMax, int strategy)
// {
//     __shared__ float tmpSum[BLOCK_DIM];
//     float tmp = 0.0f;
//     for (int id = threadIdx.x; id < n; id += BLOCK_DIM)
//     {
//         tmp += dA[id];
//     }
//     tmpSum[threadIdx.x] = tmp;
//     __syncthreads();
//     if (strategy == 0)
//     {
//         for (int step = 1; step < BLOCK_DIM; step *= 2)
//         {
//             if (threadIdx.x % (2 * step) == 0)
//             {
//                 tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
//             }
//             __syncthreads();
//         }
//         if (threadIdx.x == 0)
//         {
//             globalMax[0] = tmpSum[0];
//         }
//     }
//     else if (strategy == 1)
//     {
//         for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
//         {
//             if (threadIdx.x < step)
//             {
//                 tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
//             }
//             __syncthreads();
//         }
//         if (threadIdx.x == 0)
//         {
//             globalMax[0] = tmpSum[0];
//         }
//     }
//     else if (strategy == 2)
//     {
//         __shared__ float val[32];
//         float data = tmpSum[threadIdx.x];
//         data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
//         data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
//         data += __shfl_down_sync(0xffffffff, data, 4);
//         data += __shfl_down_sync(0xffffffff, data, 2);
//         data += __shfl_down_sync(0xffffffff, data, 1);
//         if (threadIdx.x % 32 == 0)
//         {
//             val[threadIdx.x / 32] = data;
//         }
//         __syncthreads();
//         if (threadIdx.x < 32)
//         {
//             data = val[threadIdx.x];
//             data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
//             data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
//             data += __shfl_down_sync(0xffffffff, data, 4);
//             data += __shfl_down_sync(0xffffffff, data, 2);
//             data += __shfl_down_sync(0xffffffff, data, 1);
//         }

//         __syncthreads();
//         if (threadIdx.x == 0)
//         {
//             globalMax[0] = data;
//         }
//     }
// }
// int main()
// {
//     float *hostA;
//     int n = 102400;
//     int strategy = 2;
//     int repeat = 100;
//     hostA = (float *)malloc(n * sizeof(float));
//     for (int i = 0; i < n; i++)
//     {
//         hostA[i] = (i % 10) * 1e-1;
//     }
//     float hostMax;
//     double st, ela;
//     st = get_walltime();

//     float *dA, *globalMax;
//     cudaMalloc((void **)&dA, n * sizeof(float));
//     cudaMalloc((void **)&globalMax, sizeof(float));
//     cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     float ker_time = 0;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);
//     int BLOCK_DIM = 1024;
//     int num_block_x = n / BLOCK_DIM;
//     int num_block_y = 1;
//     dim3 grid_dim(num_block_x, num_block_y, 1);
//     dim3 block_dim(BLOCK_DIM, 1, 1);
//     for (int i = 0; i < repeat; i++)
//     {
//         addKernel<1024><<<grid_dim, block_dim>>>(dA, n, globalMax, strategy);
//     }

//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);

//     cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
//     cudaMemcpy(&hostMax, globalMax, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(dA);
//     cudaFree(globalMax);
//     ela = 1000 * (get_walltime() - st);
//     printf("n = %d: strategy:%d, GPU use time:%.4f ms, kernel time:%.4f ms\n", n, strategy, ela, ker_time / repeat);
//     printf("CPU sum:%.2f, GPU sum:%.2f\n", addCpu(hostA, n), hostMax);
//     free(hostA);

//     return 0;
// }