#pragma once
#include <stddef.h>
#include <string>

namespace curoy{
class Filter
{
    public:
        Filter();
        Filter(std::string);

        double* loFilterCoeff;
        double* hiFilterCoeff;
        size_t length;
};
}