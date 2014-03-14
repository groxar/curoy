#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
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

static std::chrono::time_point<std::chrono::high_resolution_clock> start;
void startChrono(){
	start = std::chrono::system_clock::now();
}
void timeChrono(const string message){
	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
	cout << message <<": "<< elapsed_seconds.count() <<"s"<< endl;
	start = std::chrono::high_resolution_clock::now();

}

void printGpuMem(){
	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo  (&freeMem, &totalMem);
	cout <<"Free  Gpu Mem: "<< freeMem<<endl;
	cout <<"Total Gpu Mem: "<< totalMem<<endl;
}
