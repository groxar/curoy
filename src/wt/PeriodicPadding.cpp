#include "PeriodicPadding.hpp"
#include <iostream>

namespace curoy{
    double PeriodicPadding::get(int i){

        //If the modulus in C would be mathematically correct (-1 mod 3 being 2) this would be a nicer option
        if(i < 0 && -i - 1< length)
        {
            return data[length - i];
        }
        else if(i >= 0 && i < length)
        {
            return data[i];
        }
        else if(i >= length && i < 2 * length)
        {
            return data[i - length];
        }
        else{
            std::cout << "Error: filter longer than signal" << "i: " << i << " length: " << length << std::endl;
            throw "invalid";
        }
    }
}
