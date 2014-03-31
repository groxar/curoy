#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/ann.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[ann]", "cuda artifical neural network"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	cuMatrix<double> X3(readFile("../ml/annData/X3Data.txt"));
	cuMatrix<double> theta1(readFile("../ml/annData/theta1.txt"));
	cuMatrix<double> theta2 = readFile("../ml/annData/theta2.txt");
	cuMatrix<double> X = readFile("../ml/annData/Xdata.txt");
	cuMatrix<double> Y = readFile("../ml/annData/Ydata.txt");
/*
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
	*/
	SECTION("predict"){
		ann myAnn(T(theta1),T(theta2));
		startChrono();
		REQUIRE((long)sum(myAnn.predict(X3))==22520);
		timeChrono("predict");
		for(int i = 0; i< 50;++i)
			myAnn.costFunction(X,Y,0);
		timeChrono("50 iteration");
		cout << myAnn.costFunction(X,Y,0)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 0");
		cout << myAnn.costFunction(X,Y,1)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 1");
		
	}
	SECTION("init"){
		ann myAnn(400,25,10);
		cout << sum(Y)<<endl;
		cout << sum(myAnn.predict(X))<<endl;
		startChrono();
		cout << myAnn.costFunction(X,Y,0)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 0");
		cout << myAnn.costFunction(X,Y,1)<<endl;
		cudaDeviceSynchronize();
		timeChrono("lambda 1");
		myAnn.gradientDescent(X,Y,0.2,1,1000);
		timeChrono("gradienDescent");
		cout << myAnn.predict(X)<<endl;
		printGpuMem();
		xMatrix<double> outMatrix;
		outMatrix<< myAnn.hTheta;
		writeFile(outMatrix,"hTheta");
		outMatrix<< myAnn.oTheta;
		writeFile(outMatrix,"oTheta");

		cudaDeviceReset();
	}
}
