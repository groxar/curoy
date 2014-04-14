#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/kmeans.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[ann]", "cuda artifical neural network"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);
	cuMatrix<double> X(readFile("../ml/kmeansData/kmeansX"));

	SECTION("init"){
		cuMatrix<double> centroids({{6.4,2.8,5.6,2.2},{6.2,2.9,4.3,1.3},{7.7,3.0,6.1,2.3}});
		kmeans myKmeans(centroids);
		myKmeans.train(X);
		cout <<myKmeans.predict(X)<<endl;
	}
}
