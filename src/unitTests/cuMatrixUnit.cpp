#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"

using namespace std;
using namespace curoy;

TEST_CASE(  "[cuMatrix]", "cuda matrix unit test"){ 
	
	int data1[4][5] = {{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{10,11,12,13,14}};
	cuMatrix<int> dMatrix;
	xMatrix<int> hMatrix((int*)data1,{4,5});
	xMatrix<int> result;
	xMatrix <double> hMatrix1({{{1,2,3},{4,5,6},{1,2,3}},{{7,8,9},{4,5,6,7,8}}});
	cuMatrix<double> dMatrix1({{{1,2,3},{4,5,6},{1,2,3}},{{7,8,9},{4,5,6,7,8}}});
	cuMatrix<double> dMatrix2({{1,2,3},{4,5,6}});
	xMatrix <double> hMatrix2({{1,2,3},{4,5,6}});


	SECTION("Device Query"){
		int deviceCount;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if(err != cudaSuccess)
			exit(EXIT_FAILURE);
		cout << "Number of Cuda-Devices: " << deviceCount << endl;

	}

	SECTION("Initialize"){
		REQUIRE(eq(hMatrix1,dMatrix1));
	}

	SECTION("Data transfer from host to gpu and back"){
		hMatrix >> dMatrix;
		result << dMatrix;
		REQUIRE(eq(result,hMatrix));
	}
	
	SECTION("io stream"){
		cout <<"whole matrix: "<< dMatrix1 << endl; 	
		cout <<"diver: " << dMatrix1[1]<<endl;
		cout <<"single value: "<< (int)dMatrix1[1]<<endl;
	}

	SECTION("operator +"){
		cuMatrix<int> addResultD;
		xMatrix<int> addResultH;
		hMatrix >> dMatrix;
		addResultD = dMatrix + dMatrix + dMatrix; 
		addResultH = hMatrix + hMatrix + hMatrix; 
		REQUIRE(eq(addResultD,addResultH));
	}

	SECTION("sum"){
		REQUIRE(sum(hMatrix1)==sum(dMatrix1));
	}

	SECTION("matrix multiplication"){
		cuMatrix<double> cuA({{1,2,3},{4,5,6}});
		cuMatrix<double> cuB({{1,4},{2,5},{3,6}});
		cuMatrix<double> cuC= mult(cuA,cuB);
		REQUIRE(eq(cuC,xMatrix<double>({{14,32},{32,77}})));
	}

	SECTION("cuda device reset"){
		cudaDeviceReset();
	}

	SECTION("Traspose"){
		cout << "transpose"<< endl;
		cout <<"mult: "<< mult(xMatrix<double>({{1,2,3,4,5,6,7,8,9,10},{11,12,13,14,15,16,17}}),T(xMatrix<double>({{1,2,3,4,5,6,7,8,9,10},{11,12,13,14,15,16,17}}))) << endl;
		cout <<"mult: "<< mult(cuMatrix<double>({{1,2,3,4,5,6,7,8,9,10},{11,12,13,14,15,16,17}}),T(cuMatrix<double>({{1,2,3,4,5,6,7,8,9,10},{11,12,13,14,15,16,17}}))) << endl;
		REQUIRE(eq(T(hMatrix2),T(dMatrix2)));
	}
}
