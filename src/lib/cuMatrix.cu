#include "cuMatrix.hu"

template<typename N>
__global__ void addDevice(N* lhs, N* rhs, N* result){
	int idx = (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x)+threadIdx.x;
	result[idx]=lhs[idx]+rhs[idx];
}

void addDev(int* lhs, int* rhs, int* result, size_t numElements){
	addDevice<<<1,numElements>>>(lhs,rhs,result);		
}
