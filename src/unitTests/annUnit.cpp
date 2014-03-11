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
	cuMatrix<double> X3 = readFile("../ml/annData/X3Data.txt");
	cuMatrix<double> theta1 = readFile("../ml/annData/theta1.txt");
	cuMatrix<double> theta2 = readFile("../ml/annData/theta2.txt");
	cuMatrix<double> X = readFile("../ml/annData/Xdata.txt");
	cuMatrix<double> Y = readFile("../ml/annData/Ydata.txt");

	SECTION("predict"){
		REQUIRE((long)sum(myAnn.predict(X3,theta1,theta2))==22520);
		startChrono();
		cout << myAnn.costFunction(X,Y,0,theta1,theta2)<<endl;
		timeChrono("lambda 0");
		cout << myAnn.costFunction(X,Y,1,theta1,theta2)<<endl;
		timeChrono("lambda 0");
	}
}
