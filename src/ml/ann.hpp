#pragma once
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"
#include "ann.hu"

using namespace std;

namespace curoy{
	cuMatrix<double>& sigmoid(cuMatrix<double>& X){
		sigmoidDev2(X.m_data,X.dim(0),X.dim(1));
		return X; 
	}
	cuMatrix<double>&& sigmoid(cuMatrix<double>&& X){
		sigmoidDev2(X.m_data,X.dim(0),X.dim(1));
		return move(X);
	}
	cuMatrix<double> sigmoidGradient(cuMatrix<double>& X){
		sigmoid(X);
		return X*(1-X);
	}
	cuMatrix<double>&& sigmoidGradient(cuMatrix<double>&& X){
		sigmoid(!X);
		return move(!X*(1-!X));
	}
	class ann{
	private:
		cuMatrix<double> hTheta;
		cuMatrix<double> oTheta;
	public:
		ann(){
		}
		//transpose thetas
		ann(size_t numFeatures, size_t numHiddenLayer, size_t numPossibleOutputs, double epsilon = 0.12):
			hTheta(vector<size_t>({numHiddenLayer,numFeatures+1}),fillMode::rnd), 
			oTheta(vector<size_t>({numPossibleOutputs,numHiddenLayer+1}),fillMode::rnd){

			!hTheta*2*epsilon-epsilon;
			!oTheta*2*epsilon-epsilon;
		}
		ann(const cuMatrix<double>& hTheta,const cuMatrix<double>& oTheta):
			hTheta(hTheta),
			oTheta(oTheta){
		}

		cuMatrix<size_t> predict(const cuMatrix<double>& X){
			cuMatrix<double> a(sigmoid(mult( 1 | X, T(hTheta))));
			a =sigmoid(mult( 1 | a, T(oTheta)));

			return get<1>(maxPos(a,1));
		} 

		double costFunction(const cuMatrix<double>& X, const cuMatrix<double>& y, const double lambda){
			size_t m = X.dim(0);
			size_t k = max(y)+1;
			double j=0;

			cuMatrix<double> mX = 1 | X;
			cuMatrix<double> yT({m,k},0);
			cuMatrix<double> xT;

			cuMatrix<double> theta1Grad(hTheta.dim(),0);
			cuMatrix<double> theta2Grad(oTheta.dim(),0);

			// cost calculation
			cuMatrix<double> z2 = mult(mX, T(hTheta));
			cuMatrix<double> a2 = 1 | sigmoid(z2);
			cuMatrix<double> a3 = sigmoid(mult(a2,T(oTheta)));//z3 inner mult
			cuMatrix<double> a1;
			cuMatrix<double> d3;
			cuMatrix<double> d2;

			projectMatrix(y.m_data,yT.m_data,m,k);
			j=(1.0/m)*sum(-yT*log(a3)-(1.0-yT)*log(1.0-a3));
			// remove Regulation of the first element
			// regulization
			j+=(lambda/(2.0*m))*(sum(hTheta^2)+sum(oTheta^2));
		
			// back propogation
			a1=mX;
			d3=a3-yT;
			d2= mult(d3,oTheta)*sigmoidGradient( 1 | z2);
			d2= d2({0,d2.dim(0)-1},{1,d2.dim(1)-1});

			theta1Grad = mult(T(d2),a1);
			theta2Grad = mult(T(d3),a2);
			// remove Regulation of the first element
			theta1Grad = (theta1Grad * (1.0/m))+(lambda/m)*hTheta;
			theta2Grad = (theta2Grad * (1.0/m))+(lambda/m)*oTheta;
			return j;
		}

	private:
	};

}
