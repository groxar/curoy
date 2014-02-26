#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	cuMatrix<double> gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, cuMatrix<double>& theta ,const double alpha,  size_t numIterations){
		size_t numDatasets = X.dim(0);	

		for(size_t i = 0; i < numIterations;++i){
			cout <<"within: "<< (cuMatrix<double>)(mult(X,theta)-y) * (cuMatrix<double>)X << endl;
			//theta = (theta - (sum(T((mult(X,theta) - y) * X), 1) * (alpha * (1/numDatasets))));
		}

		return theta;
	}
}
