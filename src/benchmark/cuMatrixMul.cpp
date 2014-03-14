#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include <chrono>
#include <string>
#include "../unitTests/catch.hpp"
#include "../lib/cuMatrix.hpp"

using namespace std;
using namespace curoy;

	
static std::chrono::time_point<std::chrono::high_resolution_clock> start;
void startChrono(){
	start = std::chrono::system_clock::now();
}
void endChrono(const string message){
	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
	cout << message <<": "<< elapsed_seconds.count() <<"s"<< endl;
	start = now();

}

TEST_CASE("[cuMatrixMul]", "bechmark"){ 
	xMatrix<double> matrix1({512,512},3);
	xMatrix<double> matrix2({512,512},4);
	cuMatrix<double> dmatrix1({512,512},3);
	cuMatrix<double> dmatrix2({512,512},4);
	SECTION("xMatrix Multiplication"){
		startChrono();
		mult(matrix1,matrix2);
		endChrono("xMatrix Multiplication");

	}
	SECTION("cuMatrix Multiplication"){
		startChrono();
		mult(dmatrix1,dmatrix2);
		endChrono("cuMatrix Multiplication");

	}
}	
