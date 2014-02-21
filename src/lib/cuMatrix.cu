#include "cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


//DEBUG ONLY
#include <iostream>
using namespace std;

void pseudoWorkAroundFunctionToInitiallizeAddDev(){
	addDev<int>(NULL,NULL,NULL,0);
	addDev<long>(NULL,NULL,NULL,0);
	addDev<float>(NULL,NULL,NULL,0);
	addDev<double>(NULL,NULL,NULL,0);
	multDev<double>(NULL,NULL,NULL,0,0,0);
	sum<double>(NULL,0);
	transposeDev<double>(NULL, NULL, 0, 0);
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
	dim3 dimGrid(CEIL_DIV(n,B_WIDTH),CEIL_DIV(m,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	matrixMultiplyKernel<<<dimGrid,dimBlock>>>(lhs,rhs,result,n,k,k,m,n,m);
}

__global__ void addReduce(double * input, double * output, size_t len) {
  
    __shared__ double partialSum[2*B_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

	//load Segments into shared memory
    start+t<len?
      partialSum[t]=input[start+t]:
      partialSum[t]=0;
    start+blockDim.x+t<len?
      partialSum[blockDim.x+t]=input[start+blockDim.x+t]:
  	  partialSum[blockDim.x+t]=0;

	//binary tree reduce
    for(unsigned int stride = blockDim.x; stride>=1; stride >>=1)
    {
      __syncthreads();
      if(t < stride)
        partialSum[t] += partialSum[t+stride];
    }

    if(t==0)
      output[blockIdx.x]=partialSum[0];
}

template<typename N>
N sum(N* X, size_t length){
	N result = 0;
	N* sumX;
	cudaMalloc((void**) &sumX,sizeof(N)* CEIL_DIV(length,B_SIZE*2));

	size_t dimSize = B_SIZE;
	size_t gridSize = CEIL_DIV(length,B_SIZE*2);

	addReduce<<<gridSize,dimSize>>>(X,sumX,length);
	

	N* hostSum;
	hostSum = (N*) malloc(gridSize*sizeof(N));

	cudaMemcpy(hostSum, sumX, sizeof(N) * gridSize,cudaMemcpyDeviceToHost);


	for(int i = 0; i < gridSize; ++i){
		result += hostSum[i];
	}

	cudaFree(sumX);
	free(hostSum);

	return result;
}

template<typename N>
__global__ void transposeSharedKernel(N* input, N* output,size_t nRows, size_t nCols){
	__shared__ N sInput[B_WIDTH][B_WIDTH];
	int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx * B_WIDTH + tx;
    int col = by * B_WIDTH + ty;
	int pos = row * nCols + col;
	int Tpos = col * nRows + row;

	if(row < nRows && col < nCols)
		sInput[tx][ty] = input[pos];
	__syncthreads();
	if(col < nRows && row < nCols)
		output[Tpos] = sInput[tx][ty];
}

template<typename N>
__global__ void transposeKernel(N* input, N* output,size_t nRows, size_t nCols){
	int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx * B_WIDTH + tx;
    int col = by * B_WIDTH + ty;
	int pos = row * nCols + col;

	if(row < nRows && col < nCols)
		output[col * nRows + row] = input[pos];
}

template<typename N>
void transposeDev(N* input, N* result, size_t nRows, size_t nCols){
	dim3 dimGrid(CEIL_DIV(nRows,B_WIDTH),CEIL_DIV(nCols,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	transposeKernel<<<dimGrid,dimBlock>>>(input,result,nRows,nCols);
}
