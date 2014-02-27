#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/featureNormalize.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[normalize]", "cuda feature normalization"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	xMatrix<double> hX;
	xMatrix<double> hXnormalized;
	hX = readFile("../ml/gradientData/gradientX.txt");
	hXnormalized =readFile("../ml/gradientData/XdataNormalized.txt");
	cuMatrix<double> X;
	cuMatrix<double> Xnormalized;
	hX >> X;
	Xnormalized =featureNormalize(X);
	hX << Xnormalized;
	cout <<"normalized X: "<< Xnormalized<< endl;
}
