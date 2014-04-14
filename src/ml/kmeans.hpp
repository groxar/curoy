#pragma once
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"
#include "tgmath.h"

using namespace std;

namespace curoy{
	class kmeans{
		private: 
			cuMatrix<double> centroids; // colwise centroids (centroid:= mean of cluster)
		public:
		kmeans(size_t numFeatures, size_t numClusters): 
			centroids({numClusters,numFeatures},fillMode::none){
		}
		cuMatrix<size_t> predict(const cuMatrix<double>& X){
		 	return get<1>(minPos(mult(X,-T(centroids))^2,1));
		}
		void train(const cuMatrix<double>& X){	
			size_t m = X.dim(0);
			size_t numCentroids = centroids.dim(0);
			cuMatrix<size_t> cXn(get<1>(minPos(mult(X,-T(centroids))^2,1)));
			cuMatrix<size_t> cX;
			
			do{
				cX = cXn;
				for(size_t i = 0; i < numCentroids; ++i)
					centroids[i] = (1/m)*sum(mult(T((cuMatrix<double>)cXn == i),X),0);
				cXn = get<1>(minPos(mult(X,-T(centroids))^2,1));
			}while(!eq(cX,cXn));
		}
	};
}
