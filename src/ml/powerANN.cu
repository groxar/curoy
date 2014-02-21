#include "powerANN.hu"


__global__ void d_powerANN(	double* X, size_t xH, size_t xW,
							double* Y, size_t yW, size_t yH,
							double* result, size_t rW, size_t rH){
}

__global__ void gradientDescend(){
}


void trainANN(	double* X, size_t xH, size_t xW,
				double* Y, size_t yH, size_t yW,
				double* result, size_t rH, size_t rW){
	//addReduce<<<CEIL_DIV((yH*yW),B_SIZE),B_SIZE>>>(Y,sumX,yH*yW);
	//cout << "sum of Y " << sumX[0] << endl << endl;
	
}
