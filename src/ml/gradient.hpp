#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	cuMatrix<double> gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, cuMatrix<double>& theta ,const double alpha,  size_t numIterations){
		size_t numDatasets = X.dim(0);	
		cuMatrix<double> extender;
		extender.resize(T(theta).dim());
		fill(extender,1);

		for(size_t i = 0; i < numIterations;++i){
			theta = theta - sum(T(mult((mult(X,theta) - y), extender) * X),1) * (alpha*(1.0/numDatasets));
		}

		return theta;
	}
}
