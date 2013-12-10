#include <cstdio>
#include <iostream>
#include <stdio.h>

static const int blockW = 16;
static const int blockS = blockW * blockW; 

using namespace std;

__global__ void savePosition(int* x, int* y){
	int xPos = blockDim.x * blockIdx.x +threadIdx.x;
	int yPos = blockDim.y * blockIdx.y +threadIdx.y;
	x[16*yPos+xPos] =xPos;
	y[16*yPos+xPos] =yPos;	
}
int main(){
	int x[blockS];
	int y[blockS];

	int* dx;
	int* dy;
	cudaMalloc((void**)&dx, blockS * sizeof(int));
	cudaMalloc((void**)&dy, blockS * sizeof(int));

	dim3 dimBlock(blockW,blockW);
	dim3 dimGrid(1,1);
	savePosition<<<dimGrid,dimBlock>>>(dx,dy);

	cudaMemcpy(x, dx, blockS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dy, blockS * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < blockS; i++){
		cout<< x[i] <<" "<<  y[i]<< endl;
	}
	return 0;
}
