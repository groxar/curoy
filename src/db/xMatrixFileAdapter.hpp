#pragma once
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include <hiredis.h>

namespace curoy{
/*This class stores xMatrices in human readable lists of strings, thus saving doubles as fixed point numbers with only 15 places*/
class xMatrixFileAdapter: public IxMatrixIOAdapter
{
	public:
		xMatrixFileAdapter();
		virtual ~xMatrixFileAdapter();
		virtual bool Save(string, const xMatrix<double>&);
		virtual xMatrix<double> Load(string);
	private:
		redisContext *m_redisContext;
};
}
