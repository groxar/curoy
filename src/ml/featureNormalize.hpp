#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
cuMatrix<double> featureNormalize(const cuMatrix<double>& X){
	size_t numDataSets = X.dim(0);
	cuMatrix<double> extender;
	extender.resize(vector<size_t>{numDataSets,1});
	fill(extender,1);
	cuMatrix<double> mu  = sum(X,0)/numDataSets;
	cuMatrix<double> std = pow(sum(pow(X - mult(extender, mu),2),0) / numDataSets ,0.5);  
	cuMatrix<double> sigma = (X-mult(extender,mu)) / mult(extender,std)	;
	return sigma;
}
}


