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
	public:
		cuMatrix<double> hTheta;
		cuMatrix<double> oTheta;

		ann(){
		}
		//transpose thetas
		ann(size_t numFeatures, size_t numHiddenLayer, size_t numPossibleOutputs, double epsilon = 0.12):
			hTheta(vector<size_t>({numFeatures+1,numHiddenLayer}),fillMode::rnd), 
			oTheta(vector<size_t>({numHiddenLayer+1,numPossibleOutputs}),fillMode::rnd){

			!hTheta*2*epsilon-epsilon;
			!oTheta*2*epsilon-epsilon;
		}
		ann(const cuMatrix<double>& hTheta,const cuMatrix<double>& oTheta):
			hTheta(hTheta),
			oTheta(oTheta){
		}

		cuMatrix<size_t> predict(const cuMatrix<double>& X){
			cuMatrix<double> a(sigmoid(mult( 1 | X, hTheta)));
			a =sigmoid(mult( 1 | a, oTheta));

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
			cuMatrix<double> z2 = mult(mX, hTheta);
			cuMatrix<double> a2 = 1 | sigmoid(z2);
			cuMatrix<double> a3 = sigmoid(mult(a2,oTheta));//z3 inner mult
			cuMatrix<double> a1;
			cuMatrix<double> d3;
			cuMatrix<double> d2;

			projectMatrix(y.m_data,yT.m_data,m,k);
			j=(1.0/m)*sum(-yT*log(a3)-(1.0-yT)*log(1.0-a3));
			cuMatrix<double> hThetaT(hTheta);
			hThetaT[0] = 0;
			cuMatrix<double> oThetaT(oTheta);
			oThetaT[0] = 0;
			j+=(lambda/(2.0*m))*(sum(hThetaT^2)+sum(oThetaT^2));
		
			// back propogation
			a1=mX;
			d3=a3-yT;
			d2= mult(d3,T(oTheta))*sigmoidGradient( 1 | z2);
			d2= d2({0,d2.dim(0)-1},{1,d2.dim(1)-1});

			theta1Grad = (1.0/m) * mult(T(a1),d2);
			theta2Grad = (1.0/m) * mult(T(a2),d3);
			theta1Grad = theta1Grad + (lambda/m)*hThetaT;
			theta2Grad = theta2Grad + (lambda/m)*oThetaT;
			return j;
		}
		tuple<cuMatrix<double>,cuMatrix<double>,double> gradientFunction(const cuMatrix<double>& X, const cuMatrix<double>& y, const double lambda){
			size_t m = X.dim(0);
			size_t k = max(y)+1;
			double j=0;

			cuMatrix<double> mX = 1 | X;
			cuMatrix<double> yT({m,k},0);
			cuMatrix<double> xT;

			cuMatrix<double> theta1Grad(hTheta.dim(),fillMode::none);
			cuMatrix<double> theta2Grad(oTheta.dim(),fillMode::none);

			// cost calculation
			cuMatrix<double> z2 = mult(mX, hTheta);
			cuMatrix<double> a2 = 1 | sigmoid(z2);
			cuMatrix<double> a3 = sigmoid(mult(a2,oTheta));//z3 inner mult
			cuMatrix<double> a1;
			cuMatrix<double> d3;
			cuMatrix<double> d2;

			projectMatrix(y.m_data,yT.m_data,m,k);
			j=(1.0/m)*sum(-yT*log(a3)-(1.0-yT)*log(1.0-a3));
			cuMatrix<double> hThetaT(hTheta);
			hThetaT[0] = 0;
			cuMatrix<double> oThetaT(oTheta);
			oThetaT[0] = 0;
			j+=(lambda/(2.0*m))*(sum(hThetaT^2)+sum(oThetaT^2));
		
			// back propogation
			a1=mX;
			d3=a3-yT;
			d2= mult(d3,T(oTheta))*sigmoidGradient( 1.0 | z2);
			d2= d2({0,d2.dim(0)-1},{1,d2.dim(1)-1});

			theta1Grad = (1.0/m) * mult(T(a1),d2) + (lambda/m)*hThetaT;
			theta2Grad = (1.0/m) * mult(T(a2),d3) + (lambda/m)*oThetaT;
			return make_tuple(theta1Grad,theta2Grad,j);
		}

		void gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, double const alpha, const double lambda,  size_t numIterations){
			size_t numDataSets= X.dim(0);

			for(size_t i = 0; i<numIterations;++i){
				auto gradient = gradientFunction(X,y,lambda);
				hTheta =hTheta- alpha * get<0>(gradient);
				oTheta =oTheta- alpha * get<1>(gradient);
				cout << get<2>(gradient)<<endl;
			}
		}
	};

}
