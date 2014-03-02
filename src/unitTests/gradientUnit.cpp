#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/gradient.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[gradient]", "cuda gradientDescent"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	xMatrix<double> hX;
	hX = readFile("../ml/gradientData/XdataNormalized.txt");
	xMatrix<double> hy;
	hy = readFile("../ml/gradientData/gradientY.txt");
	cuMatrix<double> X;
	cuMatrix<double> y;
	hX >> X;
	hy >> y;
	cuMatrix<double> theta({3,1},0);
	gradientDescent(X, y, theta , 0.01, 400);
	cout << theta<< endl;
	cout << "expected: 334302.063993 100087.116006 3673.548451"<< endl;
}
