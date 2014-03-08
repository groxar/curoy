#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	public class ann{
		double predict(const cuMatrix<double>& theta, const cuMatrix<double> X){

			// You need to return the following variables correctly 
			p = zeros(size(X, 1), 1);

			cuMatrix<double> myX = 1 | X;
			a = sigmoid(X*T(theta[0]));
			m = theta.dim(0);
			a = [ones(m, 1) a];
			a = a* T(theta[1]);
			a = sigmoid(a);

			[val, p]=max(a,[],2);
		} 

	private:

	};
}
