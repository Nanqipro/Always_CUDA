//  数组相加
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 在GPU上分配内存，用于存储三个数组d_x、d_y和d_z
    // 这些数组将用于后续的并行计算
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 将主机内存中的h_x和h_y数组复制到GPU内存
    // 这是必要的，因为CUDA计算将在GPU上执行，需要访问这些数据
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // 定义块的大小，这里选择128作为每个块处理的元素数量
    const int block_size = 128;
    
    // 根据总元素数量N和块大小计算需要的块数量
    // 这里确保所有的元素都能被分配到块中处理
    const int grid_size = N / block_size;
    
    // 调用CUDA内核函数add，执行并行计算
    // d_x, d_y是输入数组，d_z是输出数组
    // grid_size, block_size定义了执行计算的网格和块的大小
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    /**
     * 从设备内存（GPU）将计算结果复制到主机内存（CPU）。
     * 这一步是将GPU上计算得到的数据传输到CPU上，以便于后续的处理或检查。
     * @param h_z 指向主机内存中接收数据的指针。
     * @param d_z 指向设备内存中数据源的指针。
     * @param M 复制的数据数量。
     * @param cudaMemcpyDeviceToHost 表示复制的方向是从设备到主机。
     */
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    /**
     * 检查计算结果是否符合预期。
     * 此函数用于验证GPU上的计算是否正确，是调试和验证GPU代码的重要步骤。
     * @param h_z 指向主机内存中存放计算结果的指针。
     * @param N 指定检查的数据长度。
     */
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
