#pragma once
#include "../lib/xMatrix.hpp"
namespace curoy{
class IxMatrixIOAdapter
{
	public:
		virtual ~IxMatrixIOAdapter() {}
		virtual bool Save(string, const xMatrix<double>&) = 0;
		virtual xMatrix<double> Load(string) = 0;
};
}
