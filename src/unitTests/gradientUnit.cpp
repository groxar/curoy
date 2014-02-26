#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/gradient.hpp"
#include "../ml/util.hpp"

using namespace std;
using namespace curoy;

TEST_CASE("[gradient]", "cuda gradientDescent"){ 
	xMatrix<double> hX;
	hX = readFile("../ml/gradientData/gradientX.txt");
	xMatrix<double> hy;
	hy = readFile("../ml/gradientData/gradientY.txt");
	cuMatrix<double> X;
	cuMatrix<double> y;
	hX >> X;
	hy >> y;
	cuMatrix<double> theta;
	theta.resize(vector<size_t>({3,1}));
	fill(theta,1);
	cout << gradientDescent(X, y, theta , 0.01, 2)<< endl;
}
