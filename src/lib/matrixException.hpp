#pragma once
#include <exception>

class matrixDimException{
	virtual const char* what() const throw()
	{
		return "the dimensions don't fit";
	}
};

enum class memPermission{user, owner};


