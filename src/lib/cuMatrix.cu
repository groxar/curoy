#include "cuMatrix.hu"

void pseudoWorkAroundFunctionToInitiallizeAddDev(){
	addDev<int>(NULL,NULL,NULL,0);
	addDev<long>(NULL,NULL,NULL,0);
	addDev<float>(NULL,NULL,NULL,0);
	addDev<double>(NULL,NULL,NULL,0);
}

template<typename N>
__global__ void addDevice(N* lhs, N* rhs, N* result, size_t numElements){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	if(idx<numElements)
		result[idx]=lhs[idx]+rhs[idx];
}

template<typename N> 
void addDev(N* lhs, N* rhs, N* result, size_t numElements){
	addDevice<N><<<CEIL_DIV(numElements,B_SIZE),B_SIZE>>>(lhs,rhs,result,numElements);		
}
