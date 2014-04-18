#include "cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include"mapFunc.hu"
#include "posReduce.hu"
#include <limits>

//DEBUG ONLY
#include <iostream>

namespace curoy{
	//find a beatiful way for that BS

template<typename N>
void pseudoWorkAroundFunctionToInitiallizeAddDev(){
	addDev<N>(NULL,NULL,NULL,0);
	subDev<N>(NULL,NULL,NULL,0);
	mulDev<N>(NULL,NULL,NULL,0);
	divDev<N>(NULL,NULL,NULL,0);
	eqDev<N>(NULL,NULL,NULL,0);
	neqDev<N>(NULL,NULL,NULL,0);
	addSkalarDev<N>(NULL,0,NULL,0);
	subSkalarDev<N>(NULL,0,NULL,0);
	mulSkalarDev<N>(NULL,0,NULL,0);
	divSkalarDev<N>(NULL,0,NULL,0);
	divReverseSkalarDev<N>(NULL,0,NULL,0);
	eqSkalarDev<N>(NULL,0,NULL,0);
	neqSkalarDev<N>(NULL,0,NULL,0);
	multDev<N>(NULL,NULL,NULL,0,0,0);
	prodDev<N>(NULL,0);
	sumDev<N>(NULL,0);
	maxDev<N>(NULL,0);
	minDev<N>(NULL,0);
	sumColumneDev<N>(NULL,NULL,0,0);
	prodColumneDev<N>(NULL,NULL,0,0);
	maxColumneDev<N>(NULL,NULL,0,0);
	minColumneDev<N>(NULL,NULL,0,0);
	transposeDev<N>(NULL, NULL, 0, 0);
	fillDev<N>(NULL,0,0);
	fillIdentityDev<N>(NULL,0,0);

	castDev<N,size_t>(NULL,NULL,0);
	castDev<size_t,N>(NULL,NULL,0);
	//extern
	maxPosColumneDev<N>(NULL,NULL,NULL,0, 0);
	minPosColumneDev<N>(NULL,NULL,NULL,0, 0);
}

template<typename N>
void pseudoUndefinedNonFloatingPoint(){
	powDev<N>(NULL,0,NULL,0);
	expDev<N>(NULL,NULL,0);
	logDev<N>(NULL,NULL,0);
	log10Dev<N>(NULL,NULL,0);
}
void pseudoPseudoFoo(){
	pseudoWorkAroundFunctionToInitiallizeAddDev<double>();
	pseudoWorkAroundFunctionToInitiallizeAddDev<float>();
	pseudoWorkAroundFunctionToInitiallizeAddDev<size_t>();
	pseudoWorkAroundFunctionToInitiallizeAddDev<long>();
	pseudoWorkAroundFunctionToInitiallizeAddDev<int>();

	pseudoUndefinedNonFloatingPoint<double>();
	pseudoUndefinedNonFloatingPoint<float>();
}

	
/**
 * MATRIX MULTIPLICATION
 */

template<typename N>
__global__ void matrixMultiplyKernel(const N* lhs,const N* rhs, N* result,
			             size_t numARows, size_t numAColumns,
			             size_t numBRows, size_t numBColumns,
			             size_t numCRows, size_t numCColumns) {
    __shared__ N ds_A[B_WIDTH][B_WIDTH];
    __shared__ N ds_B[B_WIDTH][B_WIDTH];
    size_t by = blockIdx.x;
    size_t bx = blockIdx.y;
    size_t ty = threadIdx.x;
    size_t tx = threadIdx.y;
    
    size_t row = bx * B_WIDTH + tx;
    size_t col = by * B_WIDTH + ty;
    N pValue = 0;
	for (size_t n = 0; n < CEIL_DIV(numAColumns,B_WIDTH); ++n){

		if(row < numARows && n*B_WIDTH+ty < numAColumns)
			ds_A[tx][ty] = lhs[(row*numAColumns)+ (n*B_WIDTH+ty)];
		else
			ds_A[tx][ty] = 0;

		if(col < numBColumns&& n*B_WIDTH+tx < numBRows)
			ds_B[tx][ty] = rhs[((n*B_WIDTH+tx)*numBColumns) + col];
		else
			ds_B[tx][ty] = 0;

		__syncthreads();
		for(size_t k = 0;k < B_WIDTH; ++k){
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
__global__ void addReduce(const N* input, N* output,N neutralValue, size_t len) {
	reduceFuncKernel(&addFuncKernel<N>,input, output,neutralValue,len);	
}	

template<typename N>
inline void sumColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&addReduce<N>,X,result, (N)0, nRows, nCols);
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
__global__ void prodReduce(const N* input, N* output,N neutralValue, size_t len) {
	reduceFuncKernel(&mulFuncKernel<N>,input, output,neutralValue,len);	
}	

template<typename N>
inline void prodColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&prodReduce<N>,X,result,(N)1,nRows, nCols);
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
__global__ void maxReduce(const N* input, N* output,N neutralValue, size_t len) {
	reduceFuncKernel(&maxFuncKernel<N>,input, output,neutralValue, len);	
}	

template<typename N>
void maxColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&maxReduce<N>,X,result,std::numeric_limits<N>::min(),nRows, nCols);
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
__global__ void minReduce(const N* input, N* output,N neutralValue, size_t len) {
	reduceFuncKernel(&minFuncKernel<N>,input, output,neutralValue, len);	
}	

template<typename N>
void minColumneDev(const N* X, N* result, size_t nRows, size_t nCols){
	reduceColFunc(&minReduce<N>,X,result,std::numeric_limits<N>::max(),nRows, nCols);
}

template<typename N>
N minDev(const N* X, size_t length){
	return reduceFuncToHostValue(&minColumneDev<N>,X,length);
}


template<typename N>
__global__ void transposeSharedKernel(const N* input, N* output,size_t nRows, size_t nCols){
	__shared__ N sInput[B_WIDTH][B_WIDTH];
	size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = bx * B_WIDTH + tx;
    size_t col = by * B_WIDTH + ty;
	size_t pos = row * nCols + col;
	size_t Tpos = col * nRows + row;

	if(row < nRows && col < nCols)
		sInput[tx][ty] = input[pos];
	__syncthreads();
	if(col < nRows && row < nCols)
		output[Tpos] = sInput[tx][ty];
}

template<typename N>
__global__ void transposeKernel(const N* input, N* output,size_t nRows, size_t nCols){
	size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = bx * B_WIDTH + tx;
    size_t col = by * B_WIDTH + ty;
	size_t pos = row * nCols + col;

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
	size_t pos = blockIdx.x * B_SIZE + threadIdx.x;
	if(pos < numElements)
		X[pos]= number;
}

template<typename N>
void fillDev(N* X, const N number, size_t numElements){
	dim3 dimGrid(CEIL_DIV(numElements,B_SIZE),1,1);
	dim3 dimBlock(B_SIZE,1,1);
	fillKernel<<<dimGrid,dimBlock>>>(X,number,numElements);
}


template<typename N>
__global__ void fillIdentityKernel(N* X, size_t nRows, size_t nCols){
	size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = bx * B_WIDTH + tx;
    size_t col = by * B_WIDTH + ty;
	size_t pos = row * nCols + col;
	if(row < nRows && col < nCols)
		X[pos] = (col==row?1:0);
}
template<typename N>
void fillIdentityDev(N* X, size_t nRows, size_t nCols){
	dim3 dimGrid(CEIL_DIV(nRows,B_WIDTH),CEIL_DIV(nCols,B_WIDTH),1);
	dim3 dimBlock(B_WIDTH,B_WIDTH,1);
	fillIdentityKernel<<<dimGrid,dimBlock>>>(X,nRows,nCols);
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
	size_t pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=pow(input[pos],exponent);
}

template<typename N>
void powDev(const N* input, const N exponent,  N* result, size_t numElements){
	powKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,exponent,result,numElements);		
}

template<typename N>
__global__ void expKernel(const N* input, N* result, size_t numElements){
	size_t pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=exp(input[pos]);
}

template<typename N>
void expDev(const N* input, N* result, size_t numElements){
	expKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}

template<typename N>
__global__ void logKernel(const N* input, N* result, size_t numElements){
	size_t pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=log(input[pos]);
}

template<typename N>
void logDev(const N* input, N* result, size_t numElements){
	logKernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}

template<typename N>
__global__ void log10Kernel(const N* input, N* result, size_t numElements){
	size_t pos = B_SIZE * blockIdx.x + threadIdx.x;
	if(pos < numElements)	
		result[pos]=log10(input[pos]);
}

template<typename N>
void log10Dev(const N* input, N* result, size_t numElements){
	log10Kernel<<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(input,result,numElements);		
}
}




