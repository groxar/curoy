#pragma once
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"
#include "ann.hu"
#include "deque"


using namespace std;

namespace curoy{
	cuMatrix<double> sigmoid(const cuMatrix<double>& X){
		cuMatrix<double> result(X);
		sigmoidDev2(result.m_data,result.dim(0),result.dim(1));
		return result; 
	}
	cuMatrix<double>&& sigmoid(cuMatrix<double>&& X){
		sigmoidDev2(X.m_data,X.dim(0),X.dim(1));
		return move(X);
	}
	cuMatrix<double> sigmoidGradient(const cuMatrix<double>& X){
		cuMatrix<double> result(X);
		sigmoid(result);
		return result*(1-result);
	}
	cuMatrix<double>&& sigmoidGradient(cuMatrix<double>&& X){
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
			hiddenLayerVec.reserve(hiddenLayerSize.size());
			
			size_t prevLayerSize = numFeatures; 
			for(size_t numNeurons : hiddenLayerSize){
				hiddenLayerVec.push_back(cuMatrix<double>({prevLayerSize+1,numNeurons},fillMode::rnd)*epsilon*2-epsilon);
				prevLayerSize = numNeurons;
			}
			hiddenLayerVec.push_back(cuMatrix<double>({prevLayerSize+1,numPossibleOutputs},fillMode::rnd));
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
		tuple<deque<cuMatrix<double>>,double> gradientFunction(const cuMatrix<double>& X, const cuMatrix<double>& y, const double lambda, size_t layerPos = 0){
	

			size_t m = X.dim(0);
			// forward propogation
			cuMatrix<double> a = 1 | X;
			cuMatrix<double> z = mult(a, hiddenLayerVec[layerPos]);
			cuMatrix<double> an = sigmoid(z);
			tuple<deque<cuMatrix<double>>,double> result;

			if(layerPos+1 == hiddenLayerVec.size()){
				// cost calculation
				size_t k = max(y)+1;
				cuMatrix<double> yP({m,k},0);
				projectMatrix(y.m_data,yP.m_data,m,k);
				double j=(1.0/m)*sum((-yP*log(an))-((1.0-yP)*log(1.0-an)));

				// cost regulization
				cuMatrix<double> thetaTemp;
				for(auto layer : hiddenLayerVec){
					thetaTemp = layer;
					thetaTemp[0] = 0; 	
					j+=(lambda/(2.0*m)) * sum(thetaTemp^2);
				}
				// back propogation
				deque<cuMatrix<double>> gradientVector;
				cuMatrix<double> d = an-yP;

				// regulization
				cuMatrix<double> regulizedLayer = hiddenLayerVec[layerPos];
				regulizedLayer[0] = 0;
				gradientVector.push_front((1.0/m)*mult(T(a),d)+(lambda/m)*regulizedLayer);
				gradientVector.push_front(d);
				result = make_tuple(gradientVector,j);
			}
			else{
				result = gradientFunction(an,y,lambda, layerPos+1);
				// back propogation
				cuMatrix<double> dn = get<0>(result)[0];
				cuMatrix<double> d = mult(dn,T(hiddenLayerVec[layerPos+1]))*sigmoidGradient(1 | z);
				d = d({0,d.dim(0)-1},{1,d.dim(1)-1});

				// regulization
				cuMatrix<double> regulizedLayer = hiddenLayerVec[layerPos];
				regulizedLayer[0] = 0;
				get<0>(result)[0] = (1.0/m) * mult(T(a),d) + (lambda/m)*regulizedLayer;
				if(layerPos > 0)
					get<0>(result).push_front(d);
			}
			return result;
		}

		void gradientDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, double const alpha, const double lambda,  size_t numIterations){
			size_t numDataSets= X.dim(0);
				tuple<deque<cuMatrix<double>>,double> gradient;

			for(size_t n = 0; n<numIterations;++n){
				gradient = gradientFunction(X,y,lambda);
				for(size_t i = hiddenLayerVec.size()-1; i > 0;--i){
					hiddenLayerVec[i] = hiddenLayerVec[i] - (alpha * get<0>(gradient)[i]);
				}
				cout << get<1>(gradient)<<endl;
			}
		}
		void conjugateDescent(const cuMatrix<double>& X, const cuMatrix<double>& y, double alpha, const double lambda,  size_t numIterations){
			size_t numDataSets = X.dim(0);
			size_t numHL = hiddenLayerVec.size();

			auto gj = gradientFunction(X,y,lambda);
			auto g = get<0>(gj);
			auto gn = get<0>(gj);
			auto d = get<0>(gj);
			double j;
			
			cuMatrix<double> e; // eucliean norm
			double beta;	
			
			//line search variables
			
			for(size_t i = numHL; i --> 0;)
				d[i] = -g[i];

			cout <<"inital cost: "<< get<1>(gj) << endl;
			for(size_t n = 0; n<numIterations;++n){
				for(size_t i = numHL; i --> 0;)
					hiddenLayerVec[i] = hiddenLayerVec[i] + alpha * d[i]; 		//update theta

				gj = gradientFunction(X,y,lambda);								//get new gradient i+1
				gn = get<0>(gj);
				j=get<1>(gj);
				cout << j<< endl;
				
				for(size_t i = numHL; i --> 0;){ 
					e = gn[i]-g[i]; 
					//beta = (gn[i] * e) / (-d[i]*g[i]);							//Liu and Storey
					beta = sum(pow(gn[i],2))/sum(d[i]*e);							//Dai and Yuan
					d[i] = -g[i]+beta*d[i];
				}
				g = gn;
			}
		}
	};

}
