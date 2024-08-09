#include "error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h> 

using namespace std;

// 邻居列表的建立

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N;//numbre of atoms
const int NUM_repeat = 20;//number of repeats
const int MN = 10; //maximum number of neighbors

const real cutoff = 1.9;//in unit of Angstroms
const real cutoff2 = cutoff * cutoff;//cutoff squared

void read_xy(vector<real> &x, vector<real> &y){
    ifstream infile("xy.txt");
    string line, word;
    if(!infile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    while(getline(infile, line)){
        stringstream words(line);
        if(line.length() == 0){
            continue;
        }
        for(int i = 0; i < 2; i++){
            if(words >> word){
                if(i == 0){
                    x.push_back(stod(word));
                }
                else{
                    y.push_back(stod(word));
                }
            }
            else{
                cout << "Error reading file" << endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void print_neighbors(const int *NN, const int *NL,const bool atomic){
    ofstream outfile("neighbors.txt");
    if(!outfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    for(int i = 0; i < N; i++){
        if(NN[i] > MN){
            cout << "Error: number of neighbors exceeds maximum" << endl;
            exit(1);
        }
        outfile <<NN[i] << endl;
        for(int j = 0;j<MN;j++){
            if(j<NN[i]){
                // 根据atomic的值选择不同的计算方式，以适应矩阵的访问需求
                int tmp = atomic?NL[i*MN+j]:NL[j*N+i];
                outfile << " "<< tmp;
            }
            else{
                outfile << -1;
            }
        }
        outfile << endl;
    }
    outfile.close();
}

void __global__ find_neighbors_atomic(int * d_NN,int *d_NL,const real *d_x,const real *d_y,const int N,const real cutoff2){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;
    }
    d_NN[i] = 0;
    const real xi = d_x[i];
    const real yi = d_y[i];
    for(int j = i+1; j < N; j++){
        const real xj = d_x[j]-xi;
        const real yj = d_y[j]-yi;
        const real r2 = xj*xj + yj*yj;
        if(r2 < cutoff2){
            d_NL[i*MN+atomicAdd(&d_NN[i],1)] = j;
            d_NL[j*MN+atomicAdd(&d_NN[j],1)] = i;
           
        }
    }
}

void __global__ find_neighbors_non_atomic(int * d_NN,int *d_NL,const real *d_x,const real *d_y,const int N,const real cutoff2){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;
    }
    d_NN[i] = 0;
    int count = 0;
    const real xi = d_x[i];
    const real yi = d_y[i];
    for (int j = 0;j<N;j++){
        const real xj = d_x[j]-xi;
        const real yj = d_y[j]-yi;
        const real r2 = xj*xj + yj*yj;
        if((r2 < cutoff2)&&(i!=j)){
            d_NL[(count++)*N+i] = j;
        }
    }
    d_NN[i] = count;
}

void timing(int *d_NN,int *d_NL,const real *d_x,const real *d_y, const bool atomic){
    for(int i = 0; i < NUM_repeat; i++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        int block_size = 128;
        int grid_size = (N + block_size - 1) / block_size;

        if(atomic){
            find_neighbors_atomic<<<grid_size, block_size>>>(d_NN,d_NL,d_x,d_y,N,cutoff2);
        }
        else{
            find_neighbors_non_atomic<<<grid_size, block_size>>>(d_NN,d_NL,d_x,d_y,N,cutoff2);
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        cout << "Time for " << N << " atoms: " << elapsedTime << " ms" << endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

int main(){
    vector<real> x, y;
    read_xy(x, y);
    N = x.size();
    int mem1 = sizeof(int) * N ;
    int mem2 = sizeof(int) * N * MN;
    int mem3 = sizeof(real) * N ;

    int *d_NN, *d_NL;
    real *d_x, *d_y;

    int *h_NN = (int*) malloc(mem1);
    int *h_NL = (int*) malloc(mem2);

    CHECK(cudaMalloc((void**)&d_NN, mem1));
    CHECK(cudaMalloc((void**)&d_NL, mem2));
    CHECK(cudaMalloc((void**)&d_x, mem3));
    CHECK(cudaMalloc((void**)&d_y, mem3));
    CHECK(cudaMemcpy(d_x, x.data(), mem3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, y.data(), mem3, cudaMemcpyHostToDevice));
    cout<<"Using atomic"<<endl;
    timing(d_NN,d_NL,d_x,d_y,true);
    cout<<"Using non-atomic"<<endl;
    timing(d_NN,d_NL,d_x,d_y,false);

    CHECK(cudaMemcpy(h_NN, d_NN, mem1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_NL, d_NL, mem2, cudaMemcpyDeviceToHost));

    print_neighbors(h_NN, h_NL, true);
    print_neighbors(h_NN, h_NL, false);

    CHECK(cudaFree(d_NN));
    CHECK(cudaFree(d_NL));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_NN);
    free(h_NL);
    return 0;


}