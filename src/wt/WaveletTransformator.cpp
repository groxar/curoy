#include "WaveletTransformator.hpp"
#include <math.h>
#include <memory.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <string>

using namespace std;
namespace curoy{
    WaveletReturn* WaveletTransformator::waveletDecomposition(const double* data, size_t length, size_t level, string waveletName)
    {
        Filter filter(waveletName);

        return waveletDecomposition(data, length, level, filter);
    }

    WaveletReturn* WaveletTransformator::waveletDecomposition(const double* data, size_t length, size_t level, Filter filter)
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
            WaveletReturn *waveletReturn = oneLevelWaveletDecomposition(temp, currentLength, filter);
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

    WaveletReturn* WaveletTransformator::oneLevelWaveletDecomposition(const double* data, size_t length, Filter filter)
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

            //TODO: warum durch sqrt(2) teilen?
            result[i] = loFilterSum;// / sqrt(2.0);
            result[i + half] = hiFilterSum;// / sqrt(2.0);
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


    double* WaveletTransformator::waveletReconstruction(const double *data, vector<size_t> levelLengths, string waveletName)
    {
        Filter filter(waveletName);
        return waveletReconstruction(data, levelLengths, filter);
    }


    double* WaveletTransformator::waveletReconstruction(const double *data, vector<size_t> levelLengths, Filter filter)
    {
        vector<size_t>::iterator it = levelLengths.begin();
        size_t lastLength = *it;
        size_t currentDataIndex = lastLength;
        double *lastData = new double[lastLength];
        copy(data, data + lastLength, lastData);
        ++it;
        for(;(it + 1) != levelLengths.end(); ++it)
        {
            size_t currentLength = *it;

            assert(currentLength <= lastLength);

            double *currentLevelData = new double[2 * currentLength](); //init with zeros

            convolutionAndUpsampling(lastData, currentLevelData, filter.loFilterCoeff, lastLength, 2 * currentLength, filter.length);

            /*cout << "1st" << endl;
            for(int k = 0; k < 2 * currentLength; ++k)
            {
                cout << currentLevelData[k] << ", ";
            }
            cout << endl;*/

            convolutionAndUpsampling(data + currentDataIndex, currentLevelData, filter.hiFilterCoeff, currentLength, 2 * currentLength, filter.length);


            /*cout << "2nd" << endl;
            for(int k = 0; k < 2 * currentLength; ++k)
            {
                cout << currentLevelData[k] << ", ";
            }
            cout << endl;*/


            delete[] lastData;
            lastData = currentLevelData;
            currentLevelData = 0;

            currentDataIndex += currentLength;
            lastLength = 2 * currentLength;
        }
        size_t totalLength = *it;
        double *toReturn = new double[totalLength];
        copy(lastData, lastData + totalLength, toReturn);
        delete[] lastData;
        return toReturn;
    }

    void WaveletTransformator::convolutionAndUpsampling(const double *data, double* out, double* filterCoeff, size_t inputLength, size_t outLength, size_t filterLength)
    {
        //Only filters with even length are allowed
        assert(filterLength % 2 == 0);

        size_t f_2 = filterLength >> 1;
        const double* dataPtr = data + f_2 - 1;

        for(size_t i = 0; i < inputLength -(f_2 - 1); ++i)
        {
            double sumEven = 0;
            double sumOdd = 0;

            for(int j = 0; j < f_2; ++j)
            {
                sumEven += filterCoeff[2 * j] * dataPtr[i-j];
                sumOdd += filterCoeff[2 * j + 1] * dataPtr[i-j];
            }

            if(outLength > 2 * i + 1)
            {
                out[2 * i + 1] += sumOdd;
                out[2 * i] += sumEven;
            }
            else if(outLength > 2 * i)
            {
                out[2 * i] += sumEven;
            }
        }
    }
}
