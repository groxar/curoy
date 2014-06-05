#pragma once
#include <stddef.h>
#include <string>
#include "IPadding.hpp"

namespace curoy{
class SymetricPadding: public IPadding
{
    public:
        SymetricPadding(const double* data, size_t length): IPadding(data, length) {}
        virtual double get(int);
};
}
