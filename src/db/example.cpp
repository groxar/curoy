#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <typeinfo>

#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include "xMatrixFileAdapter.hpp"
#include "xMatrixRedisBinaryAdapter.hpp"

using namespace curoy;


int main(int argc, char **argv)
{
    IxMatrixIOAdapter *ioAdapter = new xMatrixFileAdapter();
    IxMatrixIOAdapter *redisAdapter = new xMatrixRedisBinaryAdapter("/tmp/redis.sock");
    xMatrix<double> matrixX = ioAdapter->Load("../../Xdata.txt");
    xMatrix<double> matrixY = ioAdapter->Load("../../Ydata.txt");
    redisAdapter->Save("xdata", matrixX);
    redisAdapter->Save("ydata", matrixY);
    cout << "finished" << endl;
    delete ioAdapter;
}
