#include <iostream>
#include "cuMatrix.hpp"

using namespace std;

int main(){	
	
	int data1[4][5] = {{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{10,11,12,13,14}};
	cuMatrix<int> dMatrix;
	xMatrix<int> hMatrix((int*)data1,{4,5});
	xMatrix<int> result;
	hMatrix >> dMatrix;
	result << dMatrix;

	cout << result[0][0] << endl;
	cout << result[2][3] << endl;
	cout << result[3][4] << endl;

	cuMatrix<int> addResult;
	addResult = dMatrix+dMatrix+dMatrix;
	
	result << addResult;

	cout << result[0][0] << endl;
	cout << result[2][3] << endl;
	cout << result[3][4] << endl;


	return 0;
}
