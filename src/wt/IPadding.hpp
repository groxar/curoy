#pragma once
#include <stddef.h>
#include <string>

namespace curoy{
class IPadding
{
    public:
        const double* data;
        size_t length;

        IPadding(){}
        IPadding(const double* p_data, size_t p_length)
        {
            data = p_data;
            length = p_length;
        }
        virtual double get(int) = 0;
};
}
