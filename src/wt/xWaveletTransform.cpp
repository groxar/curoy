#include "xWaveletTransform.hpp"
#include <math.h>
#include <iostream>
#include <vector>

using namespace std;
namespace curoy{
    WaveletReturn* xWaveletTransform::doWaveletTransform(const double* data, size_t length, size_t level, xFilter filter)
    {
        size_t totalLength = 0;
        vector<WaveletReturn*> waveletReturns;

        vector<size_t> returnLengths;
        WaveletReturn* toReturn = new WaveletReturn();
        toReturn->levelLengths = returnLengths;

        const double* temp = data;

        size_t currentLength = length;
        for(size_t curLevel = 0; curLevel < level; curLevel++)
        {
            WaveletReturn *waveletReturn = doOneLevelWaveletTransform(temp, currentLength, filter);
            waveletReturns.insert(waveletReturns.begin(), waveletReturn);

            temp = waveletReturn->data;

            currentLength = waveletReturn->levelLengths[0];
            totalLength += currentLength;
            if(curLevel == level - 1){
                totalLength += currentLength;
            }
        }
        toReturn->data = new double[totalLength];
        toReturn->totalLength = totalLength;
        int writtenToData = 0;
        for(int i = 0; i < waveletReturns.size(); i++)
        {
            WaveletReturn* currentWaveletReturn = waveletReturns[i];
            if(i == 0)
            {
                toReturn->levelLengths.push_back(currentWaveletReturn->levelLengths[0]);
                toReturn->levelLengths.push_back(currentWaveletReturn->levelLengths[1]);
                for(int j = 0; j < currentWaveletReturn->levelLengths[0]; ++j)
                {
                    toReturn->data[writtenToData++] = currentWaveletReturn->data[j];
                }
            }
            else if(i > 0)
            {
                toReturn->levelLengths.push_back(currentWaveletReturn->levelLengths[1]);
            }

            for(int j = currentWaveletReturn->levelLengths[0]; j < currentWaveletReturn->levelLengths[0] + currentWaveletReturn->levelLengths[1]; ++j)
            {
               toReturn->data[writtenToData++] = currentWaveletReturn->data[j];
            }
        }
        toReturn->levelLengths.push_back(length);

        for(int i = 0; i < waveletReturns.size(); i++)
        {
            delete waveletReturns[i];
        }

        return toReturn;
    }

    WaveletReturn* xWaveletTransform::doOneLevelWaveletTransform(const double* data, size_t length, xFilter filter)
    {
        size_t resultLength = ((length + length % 2) / 2 + filter.length / 2 - 1) * 2;
        double* preparedData = new double[resultLength + filter.length - 2];
        //prepare Data
        for(int i = 0; i < resultLength + filter.length - 2; i++)
        {
            int actualIndex = i - (filter.length - 2);
            if(actualIndex >= 0 && actualIndex < (int)length){
                preparedData[i] = data[actualIndex];
            }
            else if(actualIndex >= (int)length)
            {
                preparedData[i] = data[2 * length - actualIndex - 1];
            }
            else if(actualIndex < 0)
            {
                preparedData[i] = data[abs(actualIndex) - 1];
            }
        }
        double* result = new double[resultLength];
        size_t half = resultLength >> 1;

        for(int i = 0; i < half; i++)
        {
            size_t k = (i << 1);

            double loFilterSum = 0;
            double hiFilterSum = 0;
            //loop over filter length
            for(int j = 0; j < filter.length; j++)
            {
                loFilterSum += preparedData[(k + j)] * filter.loFilterCoeff[j];
                hiFilterSum += preparedData[(k + j)] * filter.hiFilterCoeff[j];
            }

            result[i] = loFilterSum / sqrt(2.0);
            result[i + half] = hiFilterSum / sqrt(2.0);
        }

        delete[] preparedData;
        WaveletReturn *waveletReturn = new WaveletReturn();
        waveletReturn->data = result;
        waveletReturn->totalLength = resultLength;
        vector<size_t> lengths;
        lengths.push_back(half);
        lengths.push_back(half);
        waveletReturn->levelLengths = lengths;

        return waveletReturn;
    }
}
