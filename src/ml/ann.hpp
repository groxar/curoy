#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	class ann{
	public:
		cuMatrix<size_t> predict(const cuMatrix<double>& X, const cuMatrix<double>& theta1, const cuMatrix<double>& theta2){
			cuMatrix<double> a = X;
			a =sigmoid(mult( 1 | a, T(theta1)));
			a =sigmoid(mult( 1 | a, T(theta2)));

			return get<1>(maxPos(a,1));
			//return cuMatrix<size_t>({1,2},3);
		} 

	private:

		cuMatrix<double>&& sigmoid(cuMatrix<double>&& X){
			return move(1/(1+exp(-!X)));
		}
	};

}
