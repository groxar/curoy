#pragma once
#include <stddef.h>

namespace curoy{
class xWaveletTransform
{
	public:
		double* doHaarWaveletTransform(const double*, size_t);
};
}
