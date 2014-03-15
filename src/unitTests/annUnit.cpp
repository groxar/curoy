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
	cuMatrix<double> X3(readFile("../ml/annData/X3Data.txt"));
	cuMatrix<double> theta1(readFile("../ml/annData/theta1.txt"));
	cuMatrix<double> theta2 = readFile("../ml/annData/theta2.txt");
	cuMatrix<double> X = readFile("../ml/annData/Xdata.txt");
	cuMatrix<double> Y = readFile("../ml/annData/Ydata.txt");

	SECTION("performance sigmoid"){
		cuMatrix<double> X1(X);
		cuMatrix<double> X2(X);
		cuMatrix<double> X3(X);
		size_t xRows = X1.dim(0);
		size_t xCols = X1.dim(1);
		startChrono();
		for(int i = 0; i < 1000;++i)
			sigmoidDev2(X1.m_data,X1.dim(0),X1.dim(1));
		cudaDeviceSynchronize();
		timeChrono("devSigmoid2: ");
		for(int i = 0; i < 1000;++i)
			sigmoidDev2(X2.m_data,xRows,xCols);
		cudaDeviceSynchronize();
		timeChrono("devSigmoid2: ");
		printGpuMem();
		for(int i = 0; i < 1000;++i)
			sigmoid(!X3);
		timeChrono("cuSigmoid: ");
		printGpuMem();
		cout << sum(X1) << endl;
		cout << sum(X2) <<endl;
		cout << sum(X3) <<endl;
	}
	SECTION("predict"){
		REQUIRE((long)sum(myAnn.predict(X3,theta1,theta2))==22520);
		startChrono();
		printGpuMem();
		cout << myAnn.costFunction(X,Y,0,theta1,theta2)<<endl;
		timeChrono("lambda 0");
		printGpuMem();
		cout << myAnn.costFunction(X,Y,1,theta1,theta2)<<endl;
		timeChrono("lambda 1");
	}
	SECTION("init"){
		cuMatrix<double> myTheta1(theta1.dim(),fillMode::rnd);
		cuMatrix<double> myTheta2(theta2.dim(),fillMode::rnd);
		double eps = 0.12;
		!myTheta1*2*eps-eps;
		!myTheta2*2*eps-eps;
		startChrono();
		printGpuMem();
		cout << myAnn.costFunction(X,Y,0,myTheta1,myTheta2)<<endl;
		timeChrono("lambda 0");
		printGpuMem();
		cout << myAnn.costFunction(X,Y,1,myTheta1,myTheta2)<<endl;
		timeChrono("lambda 1");

		cudaDeviceReset();
	}
}
