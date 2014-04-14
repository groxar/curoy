#include "cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include"mapFunc.hu"
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
	divReverseSkalarDev<double>(NULL,0,NULL,0);
	eqSkalarDev<double>(NULL,0,NULL,0);
	neqSkalarDev<double>(NULL,0,NULL,0);
	multDev<double>(NULL,NULL,NULL,0,0,0);
	prodDev<double>(NULL,0);
	sumDev<double>(NULL,0);
	maxDev<double>(NULL,0);
	minDev<double>(NULL,0);
	sumColumneDev<double>(NULL,NULL,0,0);
	prodColumneDev<double>(NULL,NULL,0,0);
	maxColumneDev<double>(NULL,NULL,0,0);
	minColumneDev<double>(NULL,NULL,0,0);
	transposeDev<double>(NULL, NULL, 0, 0);
	transposeDev<size_t>(NULL, NULL, 0, 0);
	fillDev<double>(NULL,0,0);
	powDev<double>(NULL,0,NULL,0);
	expDev<double>(NULL,NULL,0);
	logDev<double>(NULL,NULL,0);
	log10Dev<double>(NULL,NULL,0);
	castDev<double,size_t>(NULL,NULL,0);
	castDev<size_t,double>(NULL,NULL,0);
	//extern
	maxPosColumneDev<double>(NULL,NULL,NULL,0, 0);
	minPosColumneDev<double>(NULL,NULL,NULL,0, 0);
	
	fillDev<size_t>(NULL,0,0);
	sumDev<size_t>(NULL,0);
	fillDev<float>(NULL,0,0);
	multDev<float>(NULL,NULL,NULL,0,0,0);
}


	
/**
 * MATRIX MULTIPLICATION
 */

template<typename N>
__global__ void matrixMultiplyKernel(const N* lhs,const N* rhs, N* result,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    __shared__ double ds_A[B_WIDTH][B_WIDTH];
    __shared__ double ds_B[B_WIDTH][B_WIDTH];
    int by = blockIdx.x;
    int bx = blockIdx.y;
    int ty = threadIdx.x;
    int tx = threadIdx.y;
    
    int row = bx * B_WIDTH + tx;
    int col = by * B_WIDTH + ty;
    N pValue = 0;
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
	dim3 dimGrid(CEIL_DIV(m,B_WIDTH),CEIL_DIV(n,B_WIDTH));
	dim3 dimBlock(B_WIDTH,B_WIDTH);
	matrixMultiplyKernel<<<dimGrid,dimBlock>>>(lhs,rhs,result,n,k,k,m,n,m);
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
	reduceFuncKernel(&addFuncKernel<N>,input, output, (N)0,len);	
}	

template<typename N>
inline void sumColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&addReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N sumDev(const N* X, size_t length){
	return reduceFuncToHostValue(&sumColumneDev<N>,X,length);
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
	reduceFuncKernel(&mulFuncKernel<N>,input, output, (N)1,len);	
}	

template<typename N>
inline void prodColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&prodReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N prodDev(const N* X, size_t length){
	return reduceFuncToHostValue(&prodColumneDev<N>,X,length);
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
	reduceFuncKernel(&maxFuncKernel<N>,input, output, DBL_MIN, len);	
}	

template<typename N>
void maxColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&maxReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N maxDev(const N* X, size_t length){
	return reduceFuncToHostValue(&maxColumneDev<N>,X,length);
}

/**
 * MIN REDUCE
 */
template<typename N>
__device__ inline N minFuncKernel(const N lhs, const N rhs){
	return lhs<rhs? lhs : rhs;
}

template<typename N>
__global__ void minReduce(const N* input, N* output, size_t len) {
	reduceFuncKernel(&minFuncKernel<N>,input, output, DBL_MAX, len);	
}	

template<typename N>
void minColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&minReduce<N>,X,result,nRows, nCols);
}

template<typename N>
N minDev(const N* X, size_t length){
	return reduceFuncToHostValue(&minColumneDev<N>,X,length);
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
__global__ void fillKernel(N* X, const N number, size_t numElements){
	int pos = blockIdx.x * B_SIZE + threadIdx.x;
	if(pos < numElements)
		X[pos]= number;
}

template<typename N>
void fillDev(N* X, const N number, size_t numElements){
	dim3 dimGrid(CEIL_DIV(numElements,B_SIZE),1,1);
	dim3 dimBlock(B_SIZE,1,1);
	fillKernel<<<dimGrid,dimBlock>>>(X,number,numElements);
}

/**
 * CAST
 */
template<typename M, typename N>
__device__ N castFuncKernel(M value){
	return (N) value; 
}

template<typename M, typename N>
__global__ void	castKernel(const M* X, N* result, size_t numElements){
	mapFuncKernel(&castFuncKernel<M,N>,X,result,numElements);
}

template<typename M, typename N>
void castDev(const M* X, N* result, size_t numElements){
	mapFunc(&castKernel<M,N>,X,result,numElements);
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
__device__ N divReverseFuncKernel(const N lhs, const N rhs){
	return rhs / lhs;
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

template<typename N>
__global__ void divReverseSkalarKernel(const N* lhs, const N rhs, N* result, size_t numElements){
	zipFuncSkalar(&divReverseFuncKernel<N>,lhs,rhs,result,numElements);
}

template<typename N> 
void divReverseSkalarDev(const N* lhs, const N rhs, N* result, size_t numElements){
	divReverseSkalarKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
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
	int pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=pow(input[pos],exponent);
}

template<typename N>
void powDev(const N* input, const N exponent,  N* result, size_t numElements){
	powKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,exponent,result,numElements);		
}

template<typename N>
__global__ void expKernel(const N* input, N* result, size_t numElements){
	int pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=exp(input[pos]);
}

template<typename N>
void expDev(const N* input, N* result, size_t numElements){
	expKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}

template<typename N>
__global__ void logKernel(const N* input, N* result, size_t numElements){
	int pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=log(input[pos]);
}

template<typename N>
void logDev(const N* input, N* result, size_t numElements){
	logKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}

template<typename N>
__global__ void log10Kernel(const N* input, N* result, size_t numElements){
	int pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=log10(input[pos]);
}

template<typename N>
void log10Dev(const N* input, N* result, size_t numElements){
	log10Kernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}
}




