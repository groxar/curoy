#include <string>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector> 
#include <limits> // increase printed float and double precision
#include "../lib/xMatrix.hpp"
#include "../lib/cuMatrix.hpp"
#include "powerANN.hu"

using namespace std;
using namespace curoy;

xMatrix<double> readFile(string path){
	ifstream is(path);
	double* data_ptr;
	int number_of_rows = 0; 
	int number_of_cols = 0; 
	size_t position = 0;
    string line;
	stringstream lineStream;
	string cell;
	vector<double> vf;
    
	while (std::getline(is, line)){
		lineStream << line;

		getline(lineStream,cell,' ') ;//skip first whitespace
		while (getline(lineStream,cell,' '))
		{
			vf.push_back(stod(cell));
			if(number_of_rows == 0)
				++number_of_cols;
		}
        ++number_of_rows;
		lineStream.clear();
	}
	data_ptr = (double*) malloc(number_of_cols * number_of_rows * sizeof(double));
	for(size_t i = 0; i < number_of_rows *number_of_cols; i++)
		data_ptr[i]=vf[i];

	return xMatrix<double>(data_ptr,vector<size_t>({(size_t)number_of_rows,(size_t)number_of_cols}),memPermission::owner);
}

int main(void){
	xMatrix<double> X;
	xMatrix<double> Y;
	xMatrix<double> tMatrix;
	X = readFile("Xdata.txt");
	Y = readFile("Ydata.txt");

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
	cout << "sumDim(0): " << sum(cuX,0)<<endl;

	double sumY = sum(cuY);
	cout << "sum of Y: " << sumY << endl;
	cout <<	"sum T(cuY)"<<sum(T(cuY))<< endl;
	cout << "expected: 27500"<< endl;
	//cout << "sumDim(1): " << sum(T(sum(cuY,1)),1)<<endl;
	cout << "sumDim(0): " << sum(cuY,0)<<endl;
	

	cout <<"mult sum "<< sum(mult(T(cuY),cuY))<<endl;
	cout <<"mult sum "<< sum(mult(T(Y),Y))<<endl;
	//cout <<"mutl sum"<< sum(cuX*T(cuX))<<endl;
	cout << cuX.size()*sizeof(double)<< endl;

	return EXIT_SUCCESS;
}
