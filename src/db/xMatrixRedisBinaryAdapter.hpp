#pragma once
#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include <hiredis.h>

namespace curoy{
/*This class can store xMatrices as binary data*/
class xMatrixRedisBinaryAdapter: public IxMatrixIOAdapter
{
	public:
		xMatrixRedisBinaryAdapter(string socketpath);
		virtual ~xMatrixRedisBinaryAdapter();
		virtual bool Save(string, const xMatrix<double>&);
		virtual xMatrix<double> Load(string);
	private:
		redisContext *m_redisContext;
};
}
