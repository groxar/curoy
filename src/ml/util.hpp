#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "../lib/xMatrix.hpp"
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