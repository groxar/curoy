#pragma once
#include <stddef.h>

namespace curoy{
class xFilter
{
    public:
        double* loFilterCoeff;
        double* hiFilterCoeff;
        size_t length;
};
}
