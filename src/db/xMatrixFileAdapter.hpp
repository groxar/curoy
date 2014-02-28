#pragma once
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"

namespace curoy{
class xMatrixFileAdapter: public IxMatrixIOAdapter
{
	public:
		virtual void Save(string, const xMatrix<double>&);
		virtual xMatrix<double> Load(string);
};
}
