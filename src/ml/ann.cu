#include "ann.hu"
#include "../lib/cuMatrix.hu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>

namespace curoy{


__global__ void projectMatrixKernel(const double* yVector, double* yMatrix, size_t numDataSets, size_t maxValue){
	int pos = B_SIZE*blockIdx.x+threadIdx.x;

	if(pos < numDataSets){
		size_t value = (size_t)yVector[pos];
		yMatrix[pos*maxValue+value]=1;
	}
}

void projectMatrix(const double* yVector, double* yMatrix, size_t numDataSets, size_t maxValue){
	projectMatrixKernel<<<CEIL_DIV(numDataSets,B_SIZE),B_SIZE>>>(yVector,yMatrix,numDataSets,maxValue);		
}

//stub
void costFunctionKernel(const double* X, size_t numXRows, size_t numYCols,
						const double* Y, size_t numY,
						double theta1, double theta2, //output
						size_t numHLayer){
	
}

__global__ void sigmoidKernel(const double* X, double* result,size_t numElements){
	int pos = B_SIZE*blockIdx.x+threadIdx.x;
	if(pos < numElements){
		double x = X[pos];
		result[pos] = 1/(1+exp(-1*x));
	}
}

void sigmoidDev2(double* X, size_t numRows, size_t numCols){
	sigmoidKernel<<<CEIL_DIV(numRows*numCols,B_SIZE),B_SIZE>>>(X,X,numRows*numCols);		
}

__global__ void sigmoidGradientKernel(){

}

void sigmoidDev(double* X, size_t numRows, size_t numCols){
	mulSkalarDev(X,-1.0,X,numRows*numCols);
	expDev(X,X,numRows*numCols);
	addSkalarDev(X,1.0,X,numRows*numCols);
	powDev(X,-1.0,X,numRows*numCols);
}
}
