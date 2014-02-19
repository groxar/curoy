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
	cuMatrix<int> addResult;

	SECTION("Data transfer from host to gpu and back"){
		hMatrix >> dMatrix;
		result << dMatrix;
		REQUIRE(result[0][0] == 1);
		REQUIRE(result[0][4] == 5);
		REQUIRE(result[2][3] == 10 );
		REQUIRE(result[3][0] == 10 );
		REQUIRE(result[3][4] == 14 );
	}
	
	SECTION("operator +"){
		hMatrix >> dMatrix;
		addResult = dMatrix + dMatrix + dMatrix; 
		result << addResult;
		REQUIRE(result[0][0] == 3);
		REQUIRE(result[0][4] == 15);
		REQUIRE(result[2][3] == 30 );
		REQUIRE(result[3][0] == 30 );
		REQUIRE(result[3][4] == 42 );
	}
}
