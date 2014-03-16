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
	cout << theta1.dim(0)<<endl;
	cout << theta1.dim(1)<<endl;
	cout << theta2.dim(0)<<endl;
	cout << theta2.dim(1)<<endl;

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
		timeChrono("devSigmoid2 dim");
		for(int i = 0; i < 1000;++i)
			sigmoidDev2(X2.m_data,xRows,xCols);
		cudaDeviceSynchronize();
		timeChrono("sigmoidDev2 size_t");
		for(int i = 0; i < 1000;++i)
			sigmoid(X3);
		cudaDeviceSynchronize();
		timeChrono("cuSigmoid");
		for(int i = 0; i < 1000;++i)
			sigmoid(!X3);
		cudaDeviceSynchronize();
		timeChrono("cuSigmoid move");
		for(int i = 0; i < 1000;++i)
			sigmoidGradient(X3);
		cudaDeviceSynchronize();
		timeChrono("sigmoidGradient move");
		cuMatrix<double> yt({5000,26},4);
		for(int i = 0; i < 1000;++i)
			sigmoidGradient(yt);
		cudaDeviceSynchronize();
		timeChrono("sigmoidGradient small");

		REQUIRE(sum(X1) == sum(X2));
		REQUIRE(sum(X2) == sum(X3));
		cuMatrix<double> multMat1({1000,1000},3);
		cuMatrix<double> multMat2({1000,1000},4);
		cuMatrix<double> multMat3({1000,1000},3);
		timeChrono("prepare mult test");
		mult(multMat1,multMat2);
		timeChrono("dense");
		//cout << sum(X1) <<endl;
		//cout << sum(X2) <<endl;
		//cout << sum(X3) <<endl;
	}
	SECTION("predict"){
		startChrono();
		REQUIRE((long)sum(myAnn.predict(X3,theta1,theta2))==22520);
		timeChrono("predict");
		cout << myAnn.costFunction(X,Y,0,theta1,theta2)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 0");
		cout << myAnn.costFunction(X,Y,1,theta1,theta2)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 1");
	}
	SECTION("init"){
		cuMatrix<double> myTheta1(theta1.dim(),fillMode::rnd);
		cuMatrix<double> myTheta2(theta2.dim(),fillMode::rnd);
		double eps = 0.12;
		!myTheta1*2*eps-eps;
		!myTheta2*2*eps-eps;
		startChrono();
		cout << myAnn.costFunction(X,Y,0,myTheta1,myTheta2)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 0");
		cout << myAnn.costFunction(X,Y,1,myTheta1,myTheta2)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 1");

		cudaDeviceReset();
	}
}
