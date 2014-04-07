#include "bench.hu"
#include "bench.hpp"
#include <iostream>
#include <chrono>

using namespace curoy;
using namespace std;

static std::chrono::time_point<std::chrono::high_resolution_clock> start;
void startChrono(){
    start = std::chrono::system_clock::now();
}
void endChrono(const string message){
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
    cout << message <<": "<< elapsed_seconds.count() <<"s"<< endl;
    start = std::chrono::system_clock::now();

}

int main(int argc, char **argv)
{
    double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};

    xBench xbench;
    cuBench cubench;
    startChrono();
    double* transformedData = cubench.doTransform(data, 24);
    endChrono("cudatransform");
    startChrono();
    double* transformedDataX = xbench.doTransform(data, 24);
    endChrono("xtransform");

    for(int i=0; i<24; ++i){
    	cout << transformedData[i] << " ";
    }
    cout << endl;
    for(int i=0; i<24; ++i){
        cout << transformedDataX[i] << " ";
    }
    cout << endl;
}