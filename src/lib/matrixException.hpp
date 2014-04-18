#pragma once
#include <exception>

namespace curoy{
class matrixDimException{
	virtual const char* what() const throw()
	{
		return "the dimensions don't fit";
	}
};

enum class memPermission{user, owner, diver};
enum class fillMode{rnd,none,identity};

}

