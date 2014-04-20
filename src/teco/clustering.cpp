#include "../lib/cuMatrix.hpp"
#include "../ml/kmeans.hpp"
#include "../ml/ann.hpp"
#include "../ml/util.hpp"
#include "../lib/xMatrix.hpp"
#include "../db/cuMatrixRedisBinaryAdapter.hpp"
#include <iostream>
#include <map>

using namespace std;

int main(int argc, const char *argv[]) {
	
	startChrono();
	cuMatrixRedisBinaryAdapter *redisAdapter = new cuMatrixRedisBinaryAdapter("/tmp/redis.sock");
	auto yMap = redisAdapter->LoadAll("*y:dim");
	auto xMap = redisAdapter->LoadAll("*x:dim");
	//cuMatrix<double> X({xMap.size(),24},fillMode::none);
	cuMatrix<double> X({100,24},fillMode::none);

	size_t counter = 0;
	cuMatrix<double> temp;
	cuMatrix<double> maske;
	cuMatrix<double> y;
	cuMatrix<double> extender({1,24},1);
	cuMatrix<double> mean;
	map<string,size_t> mapKeyToRow;

	timeChrono("redis load");
	for(auto entry : yMap){
		if(entry.second.dim(0)!=xMap[entry.first].dim(0))
			continue;
		temp = (cuMatrix<double>)T((cuMatrix<size_t>)xMap[entry.first])[0];
		maske = cuMatrix<double>({entry.second.dim(0),24},0);
		projectMatrix(temp.m_data,maske.m_data,entry.second.dim(0),24);
		y = mult(entry.second,extender);
		mean = sum(maske*y,0)/sum(maske,0);
		X[counter] = mean; 
		mapKeyToRow[entry.first]=counter;

		if(counter ==99)
			break;
		++counter;
	}
	timeChrono("preparation");

	cuMatrix<double> centroids({9,24},fillMode::none);
	centroids[0]=X[1];
	centroids[1]=X[2];
	centroids[2]=X[3];
	centroids[3]=X[4];
	centroids[4]=X[5];
	centroids[5]=X[6];
	centroids[6]=X[7];
	centroids[7]=X[8];
	centroids[8]=X[12];
	kmeans myKmeans(centroids);
	myKmeans.train(X);
	
	timeChrono("kmeans");

	cout << myKmeans.predict(X)<<endl;
	cudaDeviceReset();

	return 0;
}
