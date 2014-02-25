#pragma once
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include <hiredis.h>

namespace curoy{
class xMatrixRedisStringAdapter: public IxMatrixIOAdapter
{
	public:
		xMatrixRedisStringAdapter(string socketpath);
		virtual ~xMatrixRedisStringAdapter();
		virtual bool Save(string, const xMatrix<double>&);
		virtual xMatrix<double> Load(string);
	private:
		redisContext *m_redisContext;
};
}
