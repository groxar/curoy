#include "xMatrixRedisBinaryAdapter.hpp"
#include <hiredis.h>

namespace curoy{

	xMatrixRedisBinaryAdapter::xMatrixRedisBinaryAdapter(string socketpath)
	{
		m_redisContext = (redisContext *) redisConnectUnix(socketpath.c_str());
		if(m_redisContext == NULL || m_redisContext->err){
			if(m_redisContext){
				printf("Connection error: %s\n", m_redisContext->errstr);
				redisFree(m_redisContext);
			} else {
				printf("Connection error: can't allocate redis context\n");
			}
			m_redisContext = 0;
		}
	}

	xMatrixRedisBinaryAdapter::~xMatrixRedisBinaryAdapter()
	{
		redisFree(m_redisContext);
		m_redisContext = 0;
	}

	/*
	Saves an xMatrix<double> to the redis database.
	The key for the dimension information is stored at "specifiedKey:dim"
	The data itself is stored at "specifiedKey:data".
	The function returns a boolean which indicates if the matrix was stored successfully
	*/
	void xMatrixRedisBinaryAdapter::Save(string key, const xMatrix<double> &matrix)
	{
		if(m_redisContext != 0)
		{
			redisReply *reply = (redisReply *) redisCommand(m_redisContext,"DEL %s:%s", key.c_str(), "dim");
			freeReplyObject(reply);
			for(int i = 0; i < matrix.nDim(); i++){
				reply = (redisReply *) redisCommand(m_redisContext,"RPUSH %s:%s %lu", key.c_str(), "dim", matrix.dim(i));
				freeReplyObject(reply);
			}

			//calculate the length of a char array containing the matrixData
			int bytes = matrix.size() * sizeof(double) / sizeof(unsigned char);

			double *matrixData = matrix.m_data;

			reply = (redisReply *) redisCommand(m_redisContext,"DEL %s:%s", key.c_str(), "data");
			freeReplyObject(reply);
			reply = (redisReply *) redisCommand(m_redisContext, "SET %s:%s %b", key.c_str(), "data", matrixData, (size_t) bytes);
			freeReplyObject(reply);

		} else {
			throw "fehler";
		}
	}

	/*
	Loads an xMatrix from the redis database, given the key specified in the parameter.
	Throws an exception if redis cannot be reached or any other error regarding redis happens.
	*/
	xMatrix<double> xMatrixRedisBinaryAdapter::Load(string key)
	{
		if(m_redisContext != 0)
		{
			vector<size_t> dimensions;
			double* data;
			//Load Dimensions into dimensions vector
			redisReply *reply = (redisReply *) redisCommand(m_redisContext,"LRANGE %s:%s 0 -1", key.c_str(), "dim");
			if(reply->type == REDIS_REPLY_ARRAY)
			{
				
				for(int i = 0; i < reply->elements; i++)
				{
					size_t dimSize = atol(reply->element[i]->str);
					dimensions.push_back(dimSize);
				}
			}
			else{
				throw "unexpected result when loading the xMatrix dimensions";
				//TODO: error handling
			}
			freeReplyObject(reply);

			reply = (redisReply *) redisCommand(m_redisContext, "GET %s:%s", key.c_str(), "data");
			int bytes = reply->len;
			data = new double[bytes * sizeof(char) / sizeof(double)];
			memcpy(data, reply->str, bytes);
			freeReplyObject(reply);

			//create corresponding xMatrix
			xMatrix<double> matrix(data, dimensions, memPermission::owner);
			return matrix;

		} else {
			throw "redisContext is null in xMatrixRedisBinaryAdapter::Load";
			//TODO: error handling
		}
	}
}
