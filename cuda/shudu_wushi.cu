#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda.h>
#include <stdio.h>
using namespace std;

const int N = 21; // 武士数独的尺寸为21x21

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

// cpu version 检查数字是否可以放入指定位置
bool isValid(int grid[N][N], int row, int col, int num) {
    // 检查左上
    if(row < 9 && row >=0 && col <9&&col >=0){
        for(int i = 0;i< 9;i++){
            if(grid[i][col]==num)
                return false;
            if(grid[row][i]==num)
                return false;
        }
    }
    // 检查右上
    if(row < 9 && row >=0 && col <21&&col >=12){
        for(int i = 0;i< 9;i++){
            if(grid[i][col]==num)
                return false;
        }
        for(int i = 12;i< 21;i++){
            if(grid[row][i]==num)
                return false;
        }
    }
    // 检查左下
    if(row < 21 && row >=12 && col <9&&col >=0){
        for(int i = 12;i< 21;i++){
            if(grid[i][col]==num)
                return false;
        }
        for(int i =0 ;i < 9;i++){
            if(grid[row][i]==num)
                return false;
        }
    }
    // 检查右下
    if(row < 21 && row >=12 && col <21&&col >=12){
        for(int i = 12;i< 21;i++){
            if(grid[i][col]==num)
                return false;
            if(grid[row][i]==num)
                return false;
        }
    }
    // 检查中线
    if(row < 15 && row >=6 && col <15&&col >=6){
        for(int i = 6;i< 15;i++){
            if(grid[i][col]==num)
                return false;
            if(grid[row][i]==num)
                return false;
        }
        
    }
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) {
        for (int j = startCol; j < startCol + 3; j++) {
            if (grid[i][j] == num)
                return false;
        }
    }
    return true;
}

// void compare(float *hostC, float *serialC, int N)
// {
//     float error = 0;
//     bool tmp = true;
//     for (int i = 0; i < N*N; i++)
//     {
//         error = fmax(error, fabs(hostC[i] - serialC[i]));
//         if (error > 1e-5)
//         {
//             tmp = false;
//             printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
//             break;
//         }
//     }
//     if (tmp)
//     {
//         printf("GPU output all right\n");
//     }
// }

// CUDA 核函数：检查数字是否可以放入指定位置
__global__ void check_candidates(int *grid, int *candidates, int row, int col) {
    int num = threadIdx.x + 1;  // 每个线程检查一个候选数字
    bool valid = true;

    // 左上数独区域
    if (row < 9 && col < 9) {
        for (int i = 0; i < 9; i++) {
            if (grid[row * N + i] == num || grid[i * N + col] == num) {
                valid = false;
                break;
            }
        }
        int startRow = (row / 3) * 3;
        int startCol = (col / 3) * 3;
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (grid[i * N + j] == num) {
                    valid = false;
                    break;
                }
            }
        }
    }
    // 右上数独区域
    else if (row < 9 && col >= 12) {
        for (int i = 0; i < 9; i++) {
            if (grid[row * N + i + 12] == num || grid[i * N + col] == num) {
                valid = false;
                break;
            }
        }
        int startRow = (row / 3) * 3;
        int startCol = 12 + (col - 12) / 3 * 3;
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (grid[i * N + j] == num) {
                    valid = false;
                    break;
                }
            }
        }
    }
    // 左下数独区域
    else if (row >= 12 && col < 9) {
        for (int i = 12; i < 21; i++) {
            if (grid[row * N + i] == num || grid[i * N + col] == num) {
                valid = false;
                break;
            }
        }
        int startRow = 12 + (row - 12) / 3 * 3;
        int startCol = (col / 3) * 3;
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (grid[i * N + j] == num) {
                    valid = false;
                    break;
                }
            }
        }
    }
    // 右下数独区域
    else if (row >= 12 && col >= 12) {
        for (int i = 12; i < 21; i++) {
            if (grid[row * N + i] == num || grid[i * N + col] == num) {
                valid = false;
                break;
            }
        }
        int startRow = 12 + (row - 12) / 3 * 3;
        int startCol = 12 + (col - 12) / 3 * 3;
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (grid[i * N + j] == num) {
                    valid = false;
                    break;
                }
            }
        }
    }
    // 中央数独区域
    else if (row >= 6 && row < 15 && col >= 6 && col < 15) {
        for (int i = 6; i < 15; i++) {
            if (grid[row * N + i] == num || grid[i * N + col] == num) {
                valid = false;
                break;
            }
        }
        int startRow = (row / 3) * 3;
        int startCol = (col / 3) * 3;
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (grid[i * N + j] == num) {
                    valid = false;
                    break;
                }
            }
        }
    }

    // 如果数字有效，将其存入候选数组
    if (valid) {
        candidates[num - 1] = 1;
    } else {
        candidates[num - 1] = 0;
    }
}

// 求解武士数独的主机函数
bool solve_with_cuda(int *grid) {
    double st, ela;
    st = get_walltime();
    float ker_time = 0;

    int *d_grid;
    int *d_candidates;
    int candidates[9];

    // 分配设备内存
    cudaMalloc((void **)&d_grid, N * N * sizeof(int));
    cudaMalloc((void **)&d_candidates, 9 * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_grid, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);


    // 迭代遍历数独网格中的空格
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (grid[row * N + col] == 0) {  // 如果空格未填入

                cudaEvent_t start, stop;
            
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                // 启动 CUDA 线程并行检查候选数字
                // check_candidates<<<1, 9>>>(d_grid, d_candidates, row, col);
                isValid(grid, row, col);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("CUDA Error: %s\n", cudaGetErrorString(err));
                    // Possibly: exit(-1) if program cannot continue....
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

                // 复制候选结果回主机
                cudaMemcpy(candidates, d_candidates, 9 * sizeof(int), cudaMemcpyDeviceToHost);

                ela += get_walltime() - st;
                // printf("GPU use time: %.4f second\n", ela);
                printf("CPU use time: %.4f second\n", ela);

                // 在主机上处理候选结果
                for (int num = 1; num <= 9; num++) {
                    if (candidates[num - 1] == 1) {
                        // 填入候选数字，继续递归求解
                        grid[row * N + col] = num;
                        if (solve_with_cuda(grid)) {
                            cudaFree(d_grid);
                            cudaFree(d_candidates);
                            return true;
                        }
                        grid[row * N + col] = 0;  // 回溯
                    }
                }
                cudaFree(d_grid);
                cudaFree(d_candidates);
                return false; // 回溯
            }
        }
    }
    printf("CPU or GPU tiotal use time: %.4f second\n", ela);

    // 释放设备内存
    cudaFree(d_grid);
    cudaFree(d_candidates);
    return true;
}

// 打印武士数独盘面
void printGrid(int grid[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (grid[row][col] == -1) {
                cout << "  "; // 不打印无效区域
            } else {
                cout << grid[row][col] << " ";
            }
        }
        cout << endl;
    }
}

int main() {
    // 初始化武士数独网格，-1表示无效区域，0表示空位
    int grid[N][N] = {
        {9, 0, 0, 0, 5, 0, 0, 0, 7, -1, -1, -1, 2, 0, 0, 0, 9, 0, 0, 0, 6},
        {0, 0, 0, 9, 0, 7, 0, 0, 0, -1, -1, -1, 0, 0, 0, 6, 0, 3, 0, 0, 0},
        {0, 0, 0, 6, 0, 4, 0, 0, 0, -1, -1, -1, 0, 0, 0, 1, 0, 5, 0, 0, 0},
        {0, 1, 3, 0, 2, 0, 8, 9, 0, -1, -1, -1, 0, 9, 8, 0, 6, 0, 1, 5, 0},
        {2, 0, 0, 7, 0, 1, 0, 0, 3, -1, -1, -1, 6, 0, 0, 5, 0, 9, 0, 0, 8},
        {0, 9, 6, 0, 4, 0, 7, 2, 0, -1, -1, -1, 0, 7, 2, 0, 8, 0, 6, 3, 0},
        {0, 0, 0, 3, 0, 5, 0, 0, 0,  0,  0, 0, 0, 0, 0, 7, 0, 4, 0, 0, 0},
        {0, 0, 0, 4, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0},
        {3, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0,  9, 0, 0, 0, 5, 0, 0, 0, 4},
        {-1, -1, -1, -1, -1, -1, 0, 0, 0, 4, 0, 7, 0, 0, 0, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, 0, 1, 0, 0, 0, 0, 0, 8, 0, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, 0, 0, 0, 3, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1},
        {9,  0,   0,  0,  3,  0, 0, 0, 7, 0, 0, 0, 4, 0, 0,  0,  9,  0,  0,  0,  3},
        {0, 0, 0, 5, 0, 9, 0, 0, 0, 0, 5, 0, 0, 0, 0, 8, 0, 2, 0, 0, 0},
        {0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 1, 0, 0, 0},
        {0, 1, 3, 0, 2, 0, 7, 5, 0, -1, -1, -1, 0, 3, 1, 0, 7, 0, 8, 5, 0},
        {5, 0, 0, 8, 0, 7, 0, 0, 2, -1, -1, -1, 9, 0, 0, 2, 0, 3, 0, 0, 1},
        {0, 7, 4, 0, 1, 0, 9, 8, 0, -1, -1, -1, 0, 7, 5, 0, 8, 0, 3, 2, 0},
        {0, 0, 0, 6, 0, 8, 0, 0, 0, -1, -1, -1, 0, 0, 0, 4, 0, 8, 0, 0, 0},
        {0, 0, 0, 4, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 5, 0, 7, 0, 0, 0},
        {6, 0, 0, 0, 7, 0, 0, 0, 9, -1, -1, -1, 5, 0, 0, 0, 2, 0, 0, 0, 8}
    };


    printGrid(grid);
    cout << endl;

    if (solve_with_cuda((int *)grid)) {
        printGrid(grid);
    } else {
        cout << "No solution exists!" << endl;
    }

    return 0;
}
