#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include <tuple>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"

using namespace std;
using namespace curoy;

TEST_CASE("[cuMatrix]", "cuda matrix unit test"){ 
	
	double data1[4][5] = {{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{10,11,12,13,14}};
	xMatrix<double> hMatrix((double*)data1,{4,5});
	cuMatrix<double> dMatrix;
	xMatrix<double> result;
	xMatrix <double> hMatrix1({{{1,2,3},{4,5,6},{1,2,3}},{{7,8,9},{4,5,6,7,8}}});
	cuMatrix<double> dMatrix1({{{1,2,3},{4,5,6},{1,2,3}},{{7,8,9},{4,5,6,7,8}}});
	xMatrix <double> hMatrix2({{1,2,3},{4,5,6}});
	cuMatrix<double> dMatrix2({{1,2,3},{4,5,6}});


	SECTION("Device Query"){
		int deviceCount;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if(err != cudaSuccess)
			exit(EXIT_FAILURE);
		cout << "Number of Cuda-Devices: " << deviceCount << endl;

	}
	
	SECTION("Data transfer from host to gpu and back"){
		hMatrix >> dMatrix;
		result << dMatrix;
		REQUIRE(eq(result,hMatrix));
	}

	SECTION("Initialize"){
		REQUIRE(eq(hMatrix1,dMatrix1));
		cuMatrix<double> testMatrix1(vector<size_t>({2,3}),fillMode::none);
		cuMatrix<double> testMatrix2(vector<size_t>({2,3}),3.0);
		REQUIRE(dimCompare(testMatrix1.dim(),dMatrix2.dim())==0);
		fill(testMatrix1,3);
		cout <<testMatrix2<<endl;	
	}

	
	SECTION("io stream"){
		cout <<"whole matrix:"	<<endl<< dMatrix1 << endl; 	
		cout <<"diver:"			<<endl<< dMatrix1[1]<<endl;
//		cout <<"single value:"	<<endl<< (double)dMatrix1[1]<<endl;
		REQUIRE(eq(cuMatrix<double>({{1,2,3,4,5,6,7,8,9},{2,3,4,5,6,7,8,9},{3,4,5,6,7,8}})({0,2},{2,6}),cuMatrix<double>({{3,4,5,6,7},{4,5,6,7,8},{5,6,7,8,0}})));
		cout << cuMatrix<double>({4,5},fillMode::rnd)<<endl;
	}

	SECTION("Elementwise operation"){
		REQUIRE(eq(hMatrix1+3.0,dMatrix1+3.0));
		REQUIRE(eq(hMatrix1+hMatrix1,dMatrix1+dMatrix1));
		REQUIRE(eq(hMatrix1+hMatrix1+hMatrix1,dMatrix1+dMatrix1+dMatrix1));
		REQUIRE(eq(hMatrix1+(hMatrix1+hMatrix1),dMatrix1+(dMatrix1+dMatrix1)));
		REQUIRE(eq((hMatrix1+hMatrix1)+(hMatrix1+hMatrix1),(dMatrix1+dMatrix1)+(dMatrix1+dMatrix1)));
		REQUIRE(eq(hMatrix1-hMatrix1,dMatrix1-dMatrix1));
		REQUIRE(eq(hMatrix1*hMatrix1,dMatrix1*dMatrix1));
		REQUIRE(eq(hMatrix2/hMatrix2,dMatrix2/dMatrix2));
		!hMatrix1+hMatrix1;
		!dMatrix1+dMatrix1;
		REQUIRE(eq(hMatrix1,dMatrix1));
		!hMatrix2/hMatrix2;
		!dMatrix2/dMatrix2;
		REQUIRE(eq(dMatrix1,dMatrix1));
		REQUIRE(eq(dMatrix1,dMatrix1));
		REQUIRE(eq((dMatrix1==0)==(dMatrix1!=0),cuMatrix<double>(dMatrix1.dim(),0)));
		REQUIRE(eq((dMatrix1==0)!=(dMatrix1!=0),cuMatrix<double>(dMatrix1.dim(),1)));
		cout << exp(-cuMatrix<double>({{1,2,3},{4,5,6},{0,2,3}})) << endl;
		REQUIRE(eq(cuMatrix<double>({{1,3},{5,6},{3,7}}), (cuMatrix<double>) cuMatrix<size_t>({{1,3},{5,6},{3,7}})));
	}

	SECTION("Math operation"){
		REQUIRE(eq(log(dMatrix2),log(hMatrix2)));
		REQUIRE(eq(pow(dMatrix2,2),pow(hMatrix2,2)));
	}

	SECTION("sum"){
		REQUIRE(sum(hMatrix1)==sum(dMatrix1));
		REQUIRE(prod(hMatrix1)==prod(dMatrix1));
	}
	SECTION("max"){
		REQUIRE(eq(max(dMatrix2,0)[0],cuMatrix<double>({4,5,6})));
		tuple<cuMatrix<double>,cuMatrix<size_t>> maxWPos= maxPos(cuMatrix<double>({{1,2,3,4,5,6,9,1,2,3},{9,3,4,5,3,4,2},{32,41,2,4},{213241,31231414,23,4,1}}),0);
		cout << "TUPLE"<<endl;
		cout << get<0>(maxWPos)<< endl;
		cout << get<1>(maxWPos)<< endl;
	}
	SECTION("fill"){
		hMatrix << dMatrix1;
		REQUIRE(eq(fill(dMatrix1,3),fill(hMatrix1,3)));
	}

	SECTION("concationation"){
		// TODO unit test
	}

	SECTION("matrix multiplication"){
		cuMatrix<double> cuA({{1,2,3},{4,5,6}});
		cuMatrix<double> cuB({{1,4},{2,5},{3,6}});
		cuMatrix<double> cuC= mult(cuA,cuB);
		REQUIRE(eq(cuC,xMatrix<double>({{14,32},{32,77}})));
	}

	SECTION("Traspose"){
		REQUIRE(eq(T(hMatrix2),T(dMatrix2)));
	}
	
	SECTION("cuda device reset"){
		//cudaDeviceReset();
	}


}
