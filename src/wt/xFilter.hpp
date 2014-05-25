#pragma once
#include <stddef.h>
#include <string>

namespace curoy{
class xFilter
{
    public:
        xFilter();
        xFilter(std::string);

        double* loFilterCoeff;
        double* hiFilterCoeff;
        size_t length;
};
}
