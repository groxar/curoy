#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	cuMatrix gradientDescent(cuMatrix<double> X, cuMatrix<double> y, cuMatrix<double> theta,double alpha, size_t num_iters){
		size_t numDatasets = X.dim(0);	
	}
}
