#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/kmeans.hpp"
#include "../ml/util.hpp"
#include <limits>

using namespace std;
using namespace curoy;

TEST_CASE("[ann]", "cuda artifical neural network"){ 
	//increases double print precision
	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);


	SECTION("init"){
		kmeans myKmeans({400,3});
		cuMatrix<double> x({1,2,3,4,5,6});
		cuMatrix<size_t> cX({1,2,2,2,4,2});
		cuMatrix<double> z({{1,2},{3,4}});
		cout << x << endl;
		cout << cX << endl;
		cout << T(cX)<<endl;
		cout<<mult(T(x),T(T((cuMatrix<double>)cX == 2)))<<endl;
		cout << z<<endl;
		cout << z[1]<<endl;
		z[1] = cuMatrix<double>({2,23});
		cout << z<<endl;
		z[1] = 0;
		cout << z<<endl;
	}
}
