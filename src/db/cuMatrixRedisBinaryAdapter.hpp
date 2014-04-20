#pragma once
#include "../lib/cuMatrix.hpp"
#include "../lib/xMatrix.hpp"
#include <hiredis.h>
#include <map>

namespace curoy{
/*This class can store cuMatrices as binary data*/
class cuMatrixRedisBinaryAdapter
{
	public:
		cuMatrixRedisBinaryAdapter(string socketpath);
		virtual ~cuMatrixRedisBinaryAdapter();
		virtual void Save(string, const cuMatrix<double>&);
		virtual cuMatrix<double> Load(string);
		map<string,cuMatrix<double>> LoadAll(string key);
	private:
		redisContext *m_redisContext;
};
}
