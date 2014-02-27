#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	void gradientDescent(const cuMatrix<double>& X){
		size_t numDatasets = X.dim(0);	
		cuMatrix<double> extender;
		extender.resize(T(theta).dim());
		fill(extender,1);

		for(size_t i = 0; i < numIterations;++i){
			theta = theta - sum(T(mult((mult(X,theta) - y), extender) * X),1) * (alpha*(1.0/numDatasets));
		}
	}
}
