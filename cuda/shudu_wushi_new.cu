#include <iostream>
using namespace std;

const int N = 21; // 武士数独的尺寸为21x21

// 检查数字是否在行、列、3x3宫格以及交叉区域有效
bool isValid01(int grid[N][N], int row, int col, int num) {
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

__global__ void isValid02(int grid[N*N], int row, int col,int num, int &valid) {
    int grid_dim2[N][N];
    for(int i = 0;i<N*N;i++){
        grid_dim2[i/N][i%N] = grid[i];
    }

    int T_id = threadIdx.x+blockDim.x*blockIdx.x;//用9个线程判断9个位置是否有和num相等的位置
    // 检查左上
    if(row < 9 && row >=0 && col <9&&col >=0){
        if(grid_dim2[T_id][col]==num){
            valid =  0;
            return;
        }
            
        if(grid_dim2[row][T_id]==num){
            valid =  0;
            return;
        }  
    }
    // 检查右上
    if(row < 9 && row >=0 && col <21&&col >=12){
        if(grid_dim2[T_id][col]==num){
            valid =  0;
            return;
        }
        if(grid_dim2[row][T_id+12]==num){
            valid =  0;
            return;
        }
    }
    // 检查左下
    if(row < 21 && row >=12 && col <9&&col >=0){
        if(grid_dim2[T_id+12][col]==num){
            valid =  0;
            return;
        }
        if(grid_dim2[row][T_id]==num){
            valid =  0;
            return;
        }
    }
    // 检查右下
    if(row < 21 && row >=12 && col <21&&col >=12){ 
        if(grid_dim2[T_id+12][col]==num){
            valid =  0;
            return;
        }
        if(grid_dim2[row][T_id+12]==num){
            valid =  0;
            return;
        }    
    }
    // 检查中间
    if(row < 15 && row >=6 && col <15&&col >=6){
        if(grid_dim2[T_id+6][col]==num){
            valid =  0;
            return;
        }
        if(grid_dim2[row][T_id+6]==num){
            valid =  0;
            return;
        }
    }
    // 检查 3*3的方格是否满足
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) {
        for (int j = startCol; j < startCol + 3; j++) {
            if (grid_dim2[i][j] == num){
                valid =  0;
                return;
            }
        }
    }
}

// 回溯递归求解数独
bool solveSamuraiSudoku(int grid[N][N]) {
    int grid_dim1[N*N];
    for(int i = 0;i<N*N;i++){
        grid_dim1[i] = grid[i/N][i%N];
    }
    int isValid = 1;
    int *d_grid;
    cudaMalloc((void **)&d_grid, N * N * sizeof(int));
    cudaMemcpy(d_grid, grid_dim1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    // 迭代遍历数独网格中的空格
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (grid[row][col] == 0) { // 找到空位
                for (int num = 1; num <= 9; num++) {
                    printf("row: %d, col: %d, num: %d\n", row, col, num);
                    isValid02<<<3,3>>>(d_grid, row, col, num, isValid);
                    cudaDeviceSynchronize(); // 确保核函数执行完成
                    printf("isValid: %d\n",isValid);
                    if(isValid) { // 检查是否能放置数字
                        grid[row][col] = num; // 放置数字
                        if (solveSamuraiSudoku(grid)) // 递归解决下一个空位
                            cudaFree(d_grid);
                            return true;
                        grid[row][col] = 0; // 回溯
                    }
                }

                cudaFree(d_grid);
                return false; // 如果所有数字都无效，则回溯
            }
        }
    }

    cudaFree(d_grid);
    return true; // 完成所有填充
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
    if (solveSamuraiSudoku(grid)) {
        printGrid(grid);
    } else {
        cout << "No solution exists!" << endl;
    }

    return 0;
}
