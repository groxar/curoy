#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[gradient]", "cuda gradientDescent"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	SECTION("test"){
		cuMatrix<double> test({15,20},0);
		cout << test({0,test.dim(0)-1},{1,test.dim(1)-1})<<endl;
	}
}
