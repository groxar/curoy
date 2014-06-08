#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include <string>
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"
#include "../ml/util.hpp"

using namespace std;
using namespace curoy;

int main(int argc, const char *argv[]) {
	startChrono();
	cuMatrix<double> test({1,1},2); //warmup task that will take 0.5sec
	timeChrono("start");
	for(int i = 1; i <= 200;++i){
		cuMatrix<double>({i*50,i*50},fillMode::rnd);
		timeChrono("initialization");
	}
	for(int i = 1; i <= 200;++i){
		cuMatrix<double>({i*50,i*50},fillMode::rnd)+cuMatrix<double>({i*50,i*50},fillMode::rnd);
		timeChrono("addition");
	}
	for(int i = 1; i <= 200;++i){
		mult(cuMatrix<double>({i*50,i*50},fillMode::rnd),cuMatrix<double>({i*50,i*50},fillMode::rnd));
		timeChrono("matrix multiplication");
	}
	return 0;
}
	
