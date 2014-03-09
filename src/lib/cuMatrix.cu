#include "cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include "posReduce.hu"

//DEBUG ONLY
#include <iostream>
using namespace std;

namespace curoy{
void pseudoWorkAroundFunctionToInitiallizeAddDev(){
	addDev<int>(NULL,NULL,NULL,0);
	addDev<long>(NULL,NULL,NULL,0);
	addDev<float>(NULL,NULL,NULL,0);
	addDev<double>(NULL,NULL,NULL,0);
	subDev<double>(NULL,NULL,NULL,0);
	mulDev<double>(NULL,NULL,NULL,0);
	divDev<double>(NULL,NULL,NULL,0);
	eqDev<double>(NULL,NULL,NULL,0);
	neqDev<double>(NULL,NULL,NULL,0);
	addSkalarDev<double>(NULL,0,NULL,0);
	subSkalarDev<double>(NULL,0,NULL,0);
	mulSkalarDev<double>(NULL,0,NULL,0);
	divSkalarDev<double>(NULL,0,NULL,0);
	eqSkalarDev<double>(NULL,0,NULL,0);
	neqSkalarDev<double>(NULL,0,NULL,0);
	multDev<double>(NULL,NULL,NULL,0,0,0);
	prodDev<double>(NULL,0);
	sumDev<double>(NULL,0);
	maxDev<double>(NULL,0);
	sumColumneDev<double>(NULL,NULL,0,0);
	prodColumneDev<double>(NULL,NULL,0,0);
	maxColumneDev<double>(NULL,NULL,0,0);
	transposeDev<double>(NULL, NULL, 0, 0);
	fillDev<double>(NULL,0,0);
	powDev<double>(NULL,0,NULL,0);
	logDev<double>(NULL,NULL,0);
	log10Dev<double>(NULL,NULL,0);
	//exetern
	maxPosColumneDev<double>(NULL,NULL,NULL,0, 0);
}

/**
 * MATRIX MULTIPLICATION
 */
__global__ void matrixMultiplyKernel(const double * lhs,const double * rhs, double * result,
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
			ds_A[tx][ty] = lhs[(row*numAColumns)+ (n*B_WIDTH+ty)];
		else
			ds_A[tx][ty] = 0;

		if(col < numBColumns&& n*B_WIDTH+tx < numBRows)
			ds_B[tx][ty] = rhs[((n*B_WIDTH+tx)*numBColumns) + col];
		else
			ds_B[tx][ty] = 0;

		__syncthreads();
		for(int k = 0;k < B_WIDTH; ++k){
			pValue += ds_A[tx][k] * ds_B[k][ty];
		}
		__syncthreads();
	}
	if(row < numCRows && col < numCColumns){
		result[row*numCColumns+col] = pValue;
	}
}

template<typename N>
void multDev(const N* lhs, const N* rhs, N* result, size_t n, size_t k, size_t m){
	dim3 dimGrid(CEIL_DIV(n,B_WIDTH),CEIL_DIV(m,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	matrixMultiplyKernel<<<dimGrid,dimBlock>>>(lhs,rhs,result,n,k,k,m,n,m);
}

/**
 * Blockwise reduce
 * it only reduces to CEIL_DIV(len,2*B_SIZE)
 */
template<typename FUNC, typename N>
__device__ void funcReduce(FUNC f, const N* input, N* output, N neutralValue, size_t len){
	__shared__ double partialSum[2*B_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;

	//load Segments into shared memory
	start+t<len?
		partialSum[t]=input[start+t]:
		partialSum[t]=neutralValue;
	start+blockDim.x+t<len?
		partialSum[blockDim.x+t]=input[start+blockDim.x+t]:
		partialSum[blockDim.x+t]=neutralValue;

	//binary tree reduce
	for(unsigned int stride = blockDim.x; stride>=1; stride >>=1)
	{
		__syncthreads();
		if(t < stride)
			partialSum[t] =f(partialSum[t], partialSum[t+stride]);
	}

	if(t==0){
		output[blockIdx.x]=partialSum[0];
	}
}

template<typename FUNC, typename N>
void funcReduceDev(FUNC f,const N* X, N* result, size_t length){
	size_t dimSize = B_SIZE;
	size_t gridSize = CEIL_DIV(length,B_SIZE*2);
	size_t numElements;
	N* sumX;

	cudaMalloc((void**) &sumX,sizeof(N)* CEIL_DIV(length,B_SIZE*2));

	f<<<gridSize,dimSize>>>(X,sumX,length);

	while(gridSize>1){
		numElements = gridSize;
		gridSize = CEIL_DIV(gridSize,B_SIZE*2);
		f<<<gridSize,dimSize>>>(sumX,sumX,numElements);
	}
	cudaMemcpy(result,sumX,sizeof(N),cudaMemcpyDeviceToDevice);

	cudaFree(sumX);
}

template<typename FUNC, typename N>
void funcColReduceDev(FUNC f,const N* X, N* result, size_t nRows, size_t nCols){
	size_t dimSize = B_SIZE;
	size_t gridSize = CEIL_DIV(nCols,B_SIZE*2);
	size_t numElements;
	N* sumX;

	cudaMalloc((void**) &sumX,sizeof(N)* CEIL_DIV(nCols,B_SIZE*2));

	for(size_t i = 0; i < nRows;++i){
		f<<<gridSize,dimSize>>>(&(X[nCols*i]),sumX,nCols);

		while(gridSize>1){
			numElements = gridSize;
			gridSize = CEIL_DIV(gridSize,B_SIZE*2);
			f<<<gridSize,dimSize>>>(sumX,sumX,numElements);
		}
		cudaMemcpy(&(result[i]),sumX,sizeof(N),cudaMemcpyDeviceToDevice);
		gridSize = CEIL_DIV(nCols,B_SIZE*2);
	}
	cudaFree(sumX);
}

template<typename FUNC, typename N>
N funcCompleteReduceToHostValue(FUNC f, const N* X, size_t length){
	N result = 0;
	N* d_result;

	cudaMalloc((void**) &d_result,sizeof(N));
	f(X,d_result,1,length);
	
	cudaMemcpy(&result, d_result, sizeof(N), cudaMemcpyDeviceToHost);
	cudaFree(d_result);

	return result;
}

/**
 * SUM REDUCE
 */
template<typename N>
__device__ inline N addFuncKernel(const N lhs, const N rhs){
	return lhs + rhs;
}

template<typename N>
__global__ void addReduce(const N* input, N* output, size_t len) {
	funcReduce(&addFuncKernel<N>,input, output, (N)0,len);	
}	

template<typename N>
inline void sumColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	funcColReduceDev(&addReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N sumDev(const N* X, size_t length){
	return funcCompleteReduceToHostValue(&sumColumneDev<N>,X,length);
}

/**
 * PROD REDUCE
 */
template<typename N>
__device__ inline N mulFuncKernel(const N lhs, const N rhs){
	return lhs * rhs;
}

template<typename N>
__global__ void prodReduce(const N* input, N* output, size_t len) {
	funcReduce(&mulFuncKernel<N>,input, output, (N)1,len);	
}	

template<typename N>
inline void prodColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	funcColReduceDev(&prodReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N prodDev(const N* X, size_t length){
	return funcCompleteReduceToHostValue(&prodColumneDev<N>,X,length);
}

/**
 * MAX REDUCE
 */
template<typename N>
__device__ inline N maxFuncKernel(const N lhs, const N rhs){
	return lhs>rhs? lhs : rhs;
}

template<typename N>
__global__ void maxReduce(const N* input, N* output, size_t len) {
	funcReduce(&maxFuncKernel<N>,input, output, DBL_MIN, len);	
}	

template<typename N>
void maxColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	funcColReduceDev(&maxReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N maxDev(const N* X, size_t length){
	return funcCompleteReduceToHostValue(&maxColumneDev<N>,X,length);
}


template<typename N>
__global__ void transposeSharedKernel(const N* input, N* output,size_t nRows, size_t nCols){
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
__global__ void transposeKernel(const N* input, N* output,size_t nRows, size_t nCols){
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
void transposeDev(const N* input, N* result, size_t nRows, size_t nCols){
	dim3 dimGrid(CEIL_DIV(nRows,B_WIDTH),CEIL_DIV(nCols,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	transposeKernel<<<dimGrid,dimBlock>>>(input,result,nRows,nCols);
}


template<typename N>
__global__ void fillKernel(N* X, const N number, size_t size){
	int pos = blockIdx.x * B_SIZE + threadIdx.x;
	if(pos < size)
		X[pos]= number;
}

template<typename N>
void fillDev(N* X, const N number, size_t size){
	dim3 dimGrid(CEIL_DIV(size,B_SIZE),1,1);
	dim3 dimBlock(B_SIZE,1,1);
	fillKernel<<<dimGrid,dimBlock>>>(X,number,size);
}


/**
 * ZIP
 */
template<typename FUNC, typename N> 
__device__ void	zipFunc(FUNC f, const N* lhs, const N* rhs, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx<numElements)
		result[idx]=f(lhs[idx],rhs[idx]);
}

template<typename FUNC, typename N> 
__device__ void	zipFuncSkalar(FUNC f, const N* lhs, const N rhs, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx<numElements)
		result[idx]=f(lhs[idx],rhs);
}

/**
 * ADDITION
 */
template<typename N>
__global__ void addKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&addFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void addDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	addKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void addSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&addFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void addSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	addSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

/**
 * SUBTRACTION
 */
template<typename N>
__device__ N subFuncKernel(const N lhs, const N rhs){
	return lhs - rhs;
}

template<typename N>
__global__ void subKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&subFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void subDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	subKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void subSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&subFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void subSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	subSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}


/**
 * MULTIPLICATION
 */
template<typename N>
__global__ void mulKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&mulFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void mulDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	mulKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void mulSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&mulFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void mulSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	mulSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}


/**
 * DIVISION
 */
template<typename N>
__device__ N divFuncKernel(const N lhs, const N rhs){
	return lhs / rhs;
}

template<typename N>
__global__ void divKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&divFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void divDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	divKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void divSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&divFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void divSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	divSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

/**
 * Equal
 */
template<typename N>
__device__ N eqFuncKernel(const N lhs, const N rhs){
	return lhs == rhs;
}

template<typename N>
__global__ void eqKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&eqFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void eqDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	eqKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void eqSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&eqFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void eqSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	eqSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

/**
 * not Equal
 */
template<typename N>
__device__ N neqFuncKernel(const N lhs, const N rhs){
	return lhs != rhs;
}

template<typename N>
__global__ void neqKernel(const N* lhs, const N* rhs, N* result, size_t numElements){
	zipFunc(&neqFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void neqDev(const N* lhs, const N* rhs, N* result, size_t numElements){
	neqKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

template<typename N>
__global__ void neqSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&neqFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void neqSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	neqSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}

/**
 * MATH functions
 */


template<typename N>
__global__ void powKernel(const N* input,const N exponent, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx < numElements)	
		result[idx]=pow(input[idx],exponent);
}

template<typename N>
void powDev(const N* input, const N exponent,  N* result, size_t numElements){
	powKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,exponent,result,numElements);		
}

template<typename N>
__global__ void logKernel(const N* input, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx < numElements)	
		result[idx]=log(input[idx]);
}

template<typename N>
void logDev(const N* input, N* result, size_t numElements){
	logKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}

template<typename N>
__global__ void log10Kernel(const N* input, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx < numElements)	
		result[idx]=log10(input[idx]);
}

template<typename N>
void log10Dev(const N* input, N* result, size_t numElements){
	log10Kernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}
}




