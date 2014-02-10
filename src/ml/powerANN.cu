#include "powerANN.hu"


__global__ void d_powerANN(	double* X, size_t xH, size_t xW,
							double* Y, size_t yW, size_t yH,
							double* result, size_t rW, size_t rH){
}

__global__ void gradientDescend(){
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

double sum(double* X, size_t length){
	double result = 0;
	double* sumX;
	cudaMalloc((void**) &sumX,sizeof(double)* CEIL_DIV(length,B_SIZE*2));

	size_t dimSize = B_SIZE;
	size_t gridSize = CEIL_DIV(length,B_SIZE*2);

	addReduce<<<gridSize,dimSize>>>(X,sumX,length);
	

	double* hostSum;
	hostSum = (double*) malloc(gridSize*sizeof(double));

	cudaMemcpy(hostSum, sumX, sizeof(double) * gridSize,cudaMemcpyDeviceToHost);

	cout <<"gridSize "<< (int)gridSize << endl;
	cout <<"gridSize "<< length << endl;

	for(int i = 0; i < gridSize; ++i){
		result += hostSum[i];
	}

	cudaFree(sumX);
	free(hostSum);

	return result;
}


void trainANN(	double* X, size_t xH, size_t xW,
				double* Y, size_t yH, size_t yW,
				double* result, size_t rH, size_t rW){

	double sumX = sum(X,xH*xW);
	cout << "sum of X: " << sumX << endl;
	cout << "expected: 262678.260160073"<< endl;

	double sumY = sum(Y,yH*yW);
	cout << "sum of Y: " << sumY << endl;
	cout << "expected: 27500"<< endl;

	//addReduce<<<CEIL_DIV((yH*yW),B_SIZE),B_SIZE>>>(Y,sumX,yH*yW);
	//cout << "sum of Y " << sumX[0] << endl << endl;
	
}
