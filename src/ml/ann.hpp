#pragma once
#include "../lib/cuMatrix.hpp"
//#include "util.hpp"
#include "ann.hu"
#include "deque"
#include "tgmath.h"


using namespace std;

namespace curoy{
	static cuMatrix<double> sigmoid(const cuMatrix<double>& X){
		cuMatrix<double> result(X);
		sigmoidDev2(result.m_data,result.dim(0),result.dim(1));
		return result;
	}
	static cuMatrix<double>&& sigmoid(cuMatrix<double>&& X){
		sigmoidDev2(X.m_data,X.dim(0),X.dim(1));
		return move(X);
	}
	/*static cuMatrix<double> sigmoidGradient(const cuMatrix<double>& X){
		cuMatrix<double> result(X);
		sigmoid(result);
		return result*(1-result);
	}*/
	static cuMatrix<double>&& sigmoidGradient(cuMatrix<double>&& X){
		sigmoid(!X);
		return move(!X*(1-!X));
	}

	class ann{
	private:
	public:
		vector<cuMatrix<double>> hiddenLayerVec;

		ann(){
		}
		//transpose thetas
		ann(size_t numFeatures, size_t numPossibleOutputs, vector<size_t> hiddenLayerSize, double epsilon = 0.12){
			hiddenLayerVec.reserve(hiddenLayerSize.size()+1);

			size_t prevLayerSize = numFeatures;
			for(size_t numNeurons : hiddenLayerSize){
				hiddenLayerVec.push_back(cuMatrix<double>({prevLayerSize+1,numNeurons},fillMode::rnd)*epsilon*2-epsilon);
				prevLayerSize = numNeurons;
			}
			hiddenLayerVec.push_back(cuMatrix<double>({prevLayerSize+1,numPossibleOutputs},fillMode::rnd)*epsilon*2-epsilon);
		}

		ann(const vector<cuMatrix<double>>& hiddenLayerVec):
			hiddenLayerVec(hiddenLayerVec){
		}

		/**
		 * Predict
		 */
		cuMatrix<size_t> predict(const cuMatrix<double>& X){
			cuMatrix<double> tempResult(X);
			for(cuMatrix<double> layer : hiddenLayerVec){
				tempResult = sigmoid(mult(1 | tempResult,layer));
			}
			return get<1>(maxPos(tempResult,1));
		}

		/**
		 * Train
		 */
		tuple<deque<cuMatrix<double>>,double> gradientFunction(const cuMatrix<double>& X, vector<cuMatrix<double>> hl, const cuMatrix<double>& y, const double lambda, size_t layerPos = 0){

			size_t m = X.dim(0);
			// forward propogation
			cuMatrix<double> a = 1 | X;
			cuMatrix<double> z = mult(a, hl[layerPos]);
			cuMatrix<double> an = sigmoid(z);
			tuple<deque<cuMatrix<double>>,double> result;
			cout << "1"<<endl;

			if(layerPos+1 == hl.size()){
				// cost calculation
				size_t k = max(y)+1;
				cuMatrix<double> yP({m,k},0);
				projectMatrix(y.m_data,yP.m_data,m,k);
				double j=(1.0/m)*sum((-yP*log(an))-((1.0-yP)*log(1.0-an)));

				cout << "2"<<endl;
				// cost regulization
				cuMatrix<double> thetaTemp;
				for(auto layer : hl){
					thetaTemp = layer;
					thetaTemp[0] = 0;
					j+=(lambda/(2.0*m)) * sum(thetaTemp^2);
				}
				cout << "3"<<endl;
				// back propogation
				deque<cuMatrix<double>> gradientVector;
				cuMatrix<double> dn = an-yP;
				//cout <<"dn: "<< sum(dn,1)<<endl;

				cout << "4"<<endl;
				// regulization
				cuMatrix<double> regulizedLayer = hl[layerPos];
				regulizedLayer[0] = 0;
				gradientVector.push_front((1.0/m)*mult(T(a),dn)+(lambda/m)*regulizedLayer);
				gradientVector.push_front(dn);
				result = make_tuple(gradientVector,j);
				//cout <<"last gradient: "<< sum(gradientVector[1],1)<<endl;
			}
			else{
				result = gradientFunction(an,hl,y,lambda, layerPos+1);
				cout << "5"<<endl;
				// back propogation
				cuMatrix<double> dnn = get<0>(result)[0];
				cuMatrix<double> dn = mult(dnn,T(hl[layerPos+1]))*sigmoidGradient(1 | z);
				dn = dn({0,dn.dim(0)-1},{1,dn.dim(1)-1});

				cout << "6"<<endl;
				// regulization
				cuMatrix<double> regulizedLayer = hl[layerPos];
				regulizedLayer[0] = 0;
				get<0>(result)[0] = (1.0/m) * mult(T(a),dn) + (lambda/m)*regulizedLayer;
				if(layerPos > 0)
					get<0>(result).push_front(dn);
			}
			return result;
		}

		void gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, double const learnFactor, const double lambda,  size_t numIterations){
			tuple<deque<cuMatrix<double>>,double> gj;

			cout <<"X.dim(0): " << X.dim(0) <<endl;
			cout <<"X.dim(1): " << X.dim(1) <<endl;
			cout <<"y.dim(0): " << y.dim(0) <<endl;
			cout <<"y.dim(1): " << y.dim(1) <<endl;
			for(size_t n = 0; n<numIterations;++n){
				gj = gradientFunction(X,hiddenLayerVec,y,lambda);
				for(size_t i = 0; i < hiddenLayerVec.size();++i){
					hiddenLayerVec[i] = hiddenLayerVec[i] - (learnFactor * get<0>(gj)[i]);
				}
				cout << get<1>(gj)<<endl;
			}
		}

		void conjugateDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, const double lambda, const size_t numIterations){
			size_t numHL = hiddenLayerVec.size();

			auto gj = gradientFunction(X,hiddenLayerVec,y,lambda);
			auto g = get<0>(gj);
			auto gn = get<0>(gj);
			auto d = get<0>(gj);
			double j = get<1>(gj);
			double jn;
			cout << "inital cost: " << j << endl;

			cuMatrix<double> e; // eucliean norm
			cuMatrix<double> beta;
			//double dyBeta;

			//line search variables
			double learnFactor = 0.2;
			auto hl = hiddenLayerVec;
			auto hln= hiddenLayerVec;

			for(size_t i = 0; i<numHL;++i)
				d[i] = -gn[i];

			cout <<"inital cost: "<< get<1>(gj) << endl;
			for(size_t n = 0; n<numIterations;++n){
				for(size_t i = 0; i<numHL;++i)
					hln[i] = hl[i] + (learnFactor * d[i]); 		//update theta

				gj = gradientFunction(X,hln,y,lambda);								//get new gradient i+1
				gn = get<0>(gj);
				jn = get<1>(gj);

				cout << jn<< endl;

				//step calculation
				for(size_t i = 0; i<numHL;++i){

					e = gn[i]-g[i];
					beta = (gn[i] * e) / (-d[i]*g[i]);							//Liu and Storey
					//dyBeta = sqrt(sum(pow(gn[i],2)))/sum(d[i]*e);					//Dai and Yuan
					//beta = pow(gn[i],2)/(d[i]*e);								//Dai and Yuan Matrix
					//dyBeta = sum(e - 2*d[i]*sum(pow(e,2))/sum(d[i]*e)*(gn[i]/sum(d[i]*e)));

					d[i]  = -gn[i]+beta*d[i];
					//d[i] = -((hln[i]-hl[i])/(gn[i]-g[i]))*gn[i]; //pseudo Newton doesnt work!!

					// preparation i+1
					g[i]  =  gn[i];
					hl[i] = hln[i];
				}
				j = jn;
			}
			for(size_t i = 0; i<numHL;++i)
				hiddenLayerVec[i] = hl[i];
		}
	};

}
