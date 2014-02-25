#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <typeinfo>

#include <hiredis.h>
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include "xMatrixRedisStringAdapter.hpp"

using namespace curoy;

void print2d(xMatrix<double> matrix){
    if(matrix.nDim() == 2){
        for(int i = 0; i < matrix.dim(0); ++i){
            for(int j = 0; j < matrix.dim(1); ++j){
                printf("%f\t", (double) matrix[i][j]);
            }
            printf("\n");
        }
    }else{
        printf("not two dimensional!\n");
    }
}


int main(int argc, char **argv)
{
	double data[2][3] = {{1.39587453768237689,2,3},{4,5,6}};
	xMatrix<double> matrix((double*)data,{2,3});
	IxMatrixIOAdapter *ioAdapter = new xMatrixRedisStringAdapter("/tmp/redis.sock");
	ioAdapter->Save("test", matrix);
    xMatrix<double> matrixPtr = ioAdapter->Load("test");
    print2d(matrixPtr);
	delete ioAdapter;
}
