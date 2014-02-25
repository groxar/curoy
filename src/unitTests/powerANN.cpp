#include <string>
#include <iostream>
#include <iterator>
#include <vector> 
#include <limits> // increase printed float and double precision
#include "../lib/xMatrix.hpp"
#include "../lib/cuMatrix.hpp"
#include "../ml/util.hpp"

using namespace std;
using namespace curoy;


int main(void){
	xMatrix<double> X;
	xMatrix<double> Y;
	xMatrix<double> tMatrix;
	X = readFile("../ml/annData/Xdata.txt");
	Y = readFile("../ml/annData/Ydata.txt");

	typedef std::numeric_limits< double > dbl;
	cout.precision(dbl::digits10);

	cout << X[0][67] << endl;
	cout << X[22][67] << endl;
	cout << "Size: " << X.dim(0) << " " << X.dim(1) << endl;
	cout << "Size: " << Y.dim(0) << " " << Y.dim(1) << endl;
	cout << "Size: " << T(Y).dim(0) << " " << T(Y).dim(1) << endl;
	cout << T(T(T(X)))[67][0] << endl;

	cout << X.dim(0) << " " << X.dim(1) << endl;
	cout << Y.dim(0) << " " << Y.dim(1) << endl;

	cuMatrix<double> cuX;
	cuMatrix<double> cuY;
	cuMatrix<double> cuTheta;

	X >> cuX;
	Y >> cuY;

	//error tranfer error test
	tMatrix << cuX;

	double sumX = sum(cuX);
	cout << "sum of X: " << sumX << endl;
	cout <<	"sum T(cuX)"<<sum(T(cuX))<< endl;
	cout << "expected: 262678.260160073"<< endl;
	//cout << "sumDim(1): " << sum(T(sum(cuX,1)),1)<<endl;
	//cout << "sumDim(0): " << sum(cuX,0)<<endl;

	double sumY = sum(cuY);
	cout << "sum of Y: " << sumY << endl;
	cout <<	"sum T(cuY)"<<sum(T(cuY))<< endl;
	cout << "expected: 27500"<< endl;
	//cout << "sumDim(1): " << sum(T(sum(cuY,1)),1)<<endl;
	//cout << "sumDim(0): " << sum(cuY,0)<<endl;
	

	cout <<"mult sum "<< sum(mult(T(cuY),cuY))<<endl;
	cout <<"mult sum "<< sum(mult(T(Y),Y))<<endl;
	//cout <<"mutl sum"<< sum(cuX*T(cuX))<<endl;
	cout << cuX.size()*sizeof(double)<< endl;

	return EXIT_SUCCESS;
}
