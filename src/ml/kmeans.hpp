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
		kmeans(size_t numFeatures, size_t numClusters):centroids({numClusters,numFeatures},fillMode::none){}
		kmeans(cuMatrix<double> centroids):centroids(centroids){}

		cuMatrix<size_t> predict(const cuMatrix<double>& X){
			size_t m = X.dim(0);
			size_t numCentroids = centroids.dim(0);
			cuMatrix<double> y({numCentroids,m},fillMode::none); 

			for(size_t j = 0; j < numCentroids;++j)
					y[j]= T(sum((X-mult(cuMatrix<double>({m,1},1),T(centroids[j])))^2,1));
			return get<1>(minPos(y,0));
		}
		void train(const cuMatrix<double>& X, size_t maxNumIteration = 0){	
			size_t m = X.dim(0);
			size_t numCentroids = centroids.dim(0);
			cuMatrix<size_t> cX;
			cuMatrix<double> y({numCentroids,m},fillMode::none); 
			cuMatrix<double> c(centroids); 
			cuMatrix<double> cn(centroids.dim(),fillMode::none);

			for(size_t i = 0; !maxNumIteration || i < maxNumIteration; ++i){
				for(size_t j = 0; j < numCentroids;++j)
					y[j]= T(sum((X-mult(cuMatrix<double>({m,1},1),T(c[j])))^2,1));
				cX = get<1>(minPos(y,0));

				for(size_t j = 0; j < numCentroids; ++j)
					cn[j] = (1.0/sum(cX == (double)j))*sum(mult((cuMatrix<double>)(cX == j),X),0);
				if(eq(cn,c))
					break;
				c = cn;
			}
			centroids = cn;
		}
		cuMatrix<double> getCentroids(){
			return centroids;
		}
	};
}
