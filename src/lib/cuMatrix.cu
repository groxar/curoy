#include "cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


void pseudoWorkAroundFunctionToInitiallizeAddDev(){
	addDev<int>(NULL,NULL,NULL,0);
	addDev<long>(NULL,NULL,NULL,0);
	addDev<float>(NULL,NULL,NULL,0);
	addDev<double>(NULL,NULL,NULL,0);
	multDev<double>(NULL,NULL,NULL,0,0,0);
}

template<typename N>
__global__ void addKernel(N* lhs, N* rhs, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx<numElements)
		result[idx]=lhs[idx]+rhs[idx];
}

template<typename N> 
void addDev(N* lhs, N* rhs, N* result, size_t numElements){
	addKernel<N><<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

__global__ void matrixMultiplyKernel(double * A, double * B, double * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    __shared__ double ds_A[B_WIDTH][B_WIDTH];
    __shared__ double ds_B[B_WIDTH][B_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = bx * B_WIDTH + tx;
    int col = by * B_WIDTH + ty;
    double pValue = 0;
	for (int n = 0; n < CEIL_DIV(numAColumns,B_WIDTH); ++n){

		if(row < numARows && n*B_WIDTH+ty < numAColumns)
			ds_A[tx][ty] = A[(row*numAColumns)+ (n*B_WIDTH+ty)];
		else
			ds_A[tx][ty] = 0;

		if(col < numBColumns&& n*B_WIDTH+tx < numBRows)
			ds_B[tx][ty] = B[((n*B_WIDTH+tx)*numBColumns) + col];
		else
			ds_B[tx][ty] = 0;

		__syncthreads();
		for(int k = 0;k < B_WIDTH; ++k){
			pValue += ds_A[tx][k] * ds_B[k][ty];
		}
		__syncthreads();
	}
	if(row < numCRows && col < numCColumns){
		C[row*numCColumns+col] = pValue;
	}
}

template<typename N>
void multDev(N* lhs, N* rhs, N* result, size_t n, size_t k, size_t m){
	dim3 dimGrid(CEIL_DIV(n*m,B_WIDTH),CEIL_DIV(n*m,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	matrixMultiplyKernel<<<dimGrid,dimBlock>>>(lhs,rhs,result,n,k,k,m,n,m);
}
