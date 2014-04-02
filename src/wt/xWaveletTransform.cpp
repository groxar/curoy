#include "xWaveletTransform.hpp"

namespace curoy{
		double* xWaveletTransform::doHaarWaveletTransform(const double* data, size_t length)
		{
			double* transformedData = new double[length];
			double* temp = new double[length];

			for(size_t i = 0; i < length; ++i)
			{
				transformedData[i] = data[i];
			}

			for(size_t half = length >> 1; half > 0; half = half >> 1)
			{
				for (size_t i = 0; i < half; i++)
	            {
	                size_t k = (i << 1);
	                temp[i] = (transformedData[k] + transformedData[k + 1]) / 2;
	                temp[i + half] = (transformedData[k] - transformedData[k + 1]) / 2;
	            }
	            for(size_t i = 0; i < 2 * half; i++)
	            {
	            	transformedData[i] = temp[i];
	            }
			}
			return transformedData;
		}
}
