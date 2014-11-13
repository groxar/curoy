#pragma once
#include "../lib/cuMatrix.hpp"


namespace curoy{
  using namespace std;


	//TODO FUNCTIONIZE
	/*
 	void gradientDescentAnn(const cuMatrix<double>& X, const cuMatrix<double>& y, ann& myAnn,const double alpha, const double lambda,  size_t numIterations){
		size_t numDataSets= X.dim(0);

		for(size_t i = 0; i<numIterations;++i){
			auto gradient= myAnn.gradientFunction(X,y,lambda);
			myAnn.hTheta -= alpha * get<0>(gradient);
			myAnn.oTheta -= alpha * get<1>(gradient);
		}
	}
	*/

	//logistic or linear regression; TODO look up what it was!!
 	void gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, cuMatrix<double>& theta ,const double alpha,  size_t numIterations){
		size_t numDataSets = X.dim(0);

		for(size_t i = 0; i < numIterations;++i){
			theta = theta - sum(T(mult((mult(X,theta) - y), cuMatrix<double>({T(theta).dim()},1)) * X),1) * (alpha*(1.0/numDataSets));
		}
	}
}
