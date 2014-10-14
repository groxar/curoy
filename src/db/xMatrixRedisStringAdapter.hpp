#pragma once
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include <hiredis.h>

namespace curoy{
/*This class stores xMatrices in human readable lists of strings, thus saving doubles as fixed point numbers with only 15 places*/
class xMatrixRedisStringAdapter: public IxMatrixIOAdapter
{
	public:
		xMatrixRedisStringAdapter(string socketpath);
		virtual ~xMatrixRedisStringAdapter();
		virtual void Save(string, const xMatrix<double>&);
		virtual xMatrix<double> Load(string);
	private:
		redisContext *m_redisContext;
};
}
