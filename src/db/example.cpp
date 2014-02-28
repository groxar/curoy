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
    xMatrix<double> xMatrix = ioAdapter->Load("../../Xdata.txt");
    xMatrix<double> yMatrix = ioAdapter->Load("../../Ydata.txt");
    redisAdapter->Save("xdata", xMatrix);
    redisAdapter->Save("ydata", yMatrix);
    cout << "finished" << endl;
    delete ioAdapter;
}
