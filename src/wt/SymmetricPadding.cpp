#include "SymmetricPadding.hpp"
#include <iostream>

namespace curoy{
    double SymmetricPadding::get(int i){
        if(i < 0 && -i - 1< length)
        {
            return data[-i - 1];
        }
        else if(i >= 0 && i < length)
        {
            return data[i];
        }
        else if(i >= length && i < 2 * length)
        {
            return data[length - (i - length + 1)];
        }
        else{
            std::cout << "Error: filter longer than signal" << "i: " << i << " length: " << length << std::endl;
            throw "invalid";
        }
    }
}
