#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/ann.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[gradient]", "cuda gradientDescent"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	ann myAnn;
	cuMatrix<double> X = readFile("../ml/annData/X3Data.txt");
	cuMatrix<double> theta1 = readFile("../ml/annData/theta1.txt");
	cuMatrix<double> theta2 = readFile("../ml/annData/theta2.txt");

	SECTION("predict"){
		REQUIRE((int)sum(myAnn.predict(X,theta1,theta2))==22520);
	}
}
