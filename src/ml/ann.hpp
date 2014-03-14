#pragma once
#include "../lib/cuMatrix.hpp"
#include "ann.hu"

using namespace std;

namespace curoy{
	cuMatrix<double> sigmoid(cuMatrix<double>& X){
		return move(1/(1+exp(-X)));
	}
	cuMatrix<double>&& sigmoid(cuMatrix<double>&& X){
		return move(1/(1+exp(-!X)));
	}
	cuMatrix<double> sigmoidGradient(cuMatrix<double>& X){
		return sigmoid(X)*(1-sigmoid(X));
	}
	cuMatrix<double>&& sigmoidGradient(cuMatrix<double>&& X){
		return sigmoid(!X)*(1-sigmoid(!X));
	}
	class ann{
	public:
		cuMatrix<size_t> predict(const cuMatrix<double>& X, const cuMatrix<double>& theta1, const cuMatrix<double>& theta2){
			cuMatrix<double> a = X;
			a =sigmoid(mult( 1 | a, T(theta1)));
			a =sigmoid(mult( 1 | a, T(theta2)));

			return get<1>(maxPos(a,1));
			//return cuMatrix<size_t>({1,2},3);
		} 
		double costFunctionGPU(const cuMatrix<double>& X, const cuMatrix<double>& y,const double lambda, const cuMatrix<double>& theta1, const cuMatrix<double>& theta2){
			double result;
			return result;	
		}

		double costFunction(const cuMatrix<double>& X, const cuMatrix<double>& y,const double lambda, const cuMatrix<double>& theta1, const cuMatrix<double>& theta2){
			size_t m = X.dim(0);
			size_t k = max(y)+1;
			double j=0;

			cuMatrix<double> mX = 1 | X;
			cuMatrix<double> yT;
			cuMatrix<double> xT;

			cuMatrix<double> theta1Grad(theta1.dim(),0);
			cuMatrix<double> theta2Grad(theta2.dim(),0);

			// cost calculation
			cuMatrix<double> z2 = mult(mX, T(theta1));
			cuMatrix<double> a2 = 1 | sigmoid(z2);
			cuMatrix<double> z3 = mult(a2,T(theta2));
			cuMatrix<double> a3 = sigmoid(z3);
			cuMatrix<double> a1;
			cuMatrix<double> d3;
			cuMatrix<double> d2;
			for(size_t i=0; i < 1; ++i){ //m iterations
				yT = cuMatrix<double>({k},0);
				yT[~y[i]]=1;
				j+=(1.0/m)*sum(-yT*log(a3[i])-(1.0-yT)*log(1.0-a3[i]));
			}
			
			// remove Regulation of the first element
			// regulization
			j+=(lambda/(2.0*m))*(sum(theta1^2)+sum(theta2^2));
			
			for(size_t t = 0; t < 1; ++t){
				yT = cuMatrix<double>({k},0);
				yT[~y[t]]=1;
				a1=T(mX[t]);
				d3=T(a3[t]-yT);
				d2= mult(d3,theta2)*sigmoidGradient( 1 | T(z2[t]));

				d2=d2({0,0},{1,d2.dim(1)-1});
				theta1Grad = theta1Grad + mult(T(d2),a1);
				theta2Grad = theta2Grad + mult(T(d3),T(a2[t]));
			}

			// remove Regulation of the first element
			theta1Grad = (theta1Grad * (1.0/m))+(lambda/m)*theta1;
			theta2Grad = (theta2Grad * (1.0/m))+(lambda/m)*theta2;
			return j;
		}

	private:
	};

}
