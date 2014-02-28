#include "xMatrixRedisStringAdapter.hpp"
#include <hiredis.h>

namespace curoy{

	xMatrixRedisStringAdapter::xMatrixRedisStringAdapter(string socketpath)
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

	xMatrixRedisStringAdapter::~xMatrixRedisStringAdapter()
	{
		redisFree(m_redisContext);
		m_redisContext = 0;
	}

	/*
	Saves an xMatrix<double> to the redis database.
	The key for the dimension information is stored at "specifiedKey:dim"
	The data itself is stored at "specifiedKey:data".
	*/
	void xMatrixRedisStringAdapter::Save(string key, const xMatrix<double> &matrix)
	{
		if(m_redisContext != 0)
		{
			redisReply *reply = (redisReply *) redisCommand(m_redisContext,"DEL %s:%s", key.c_str(), "dim");
			freeReplyObject(reply);
			int fields = 1;
			for(int i = 0; i < matrix.nDim(); i++){
				reply = (redisReply *) redisCommand(m_redisContext,"RPUSH %s:%s %d", key.c_str(), "dim", matrix.dim(i));
				freeReplyObject(reply);
				fields *= matrix.dim(i);
			}

			double *matrixData = matrix.m_data;

			reply = (redisReply *) redisCommand(m_redisContext,"DEL %s:%s", key.c_str(), "data");
			freeReplyObject(reply);
			for(int i = 0; i < fields; i++){
				reply = (redisReply *) redisCommand(m_redisContext,"RPUSH %s:%s %#.15g", key.c_str(), "data", matrixData[i]);
				freeReplyObject(reply);
			}
		} else {
			throw "fehler";
		}
	}

	/*
	Loads an xMatrix from the redis database, given the key specified in the parameter.
	Throws an exception if redis cannot be reached or any other error regarding redis happens.
	*/
	xMatrix<double> xMatrixRedisStringAdapter::Load(string key)
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
					size_t dimSize = static_cast<size_t>(atoi(reply->element[i]->str));
					dimensions.push_back(dimSize);
				}
			}
			else{
				throw 0;
				//TODO: error handling
			}
			freeReplyObject(reply);

			//Load Data into array of doubles
			reply = (redisReply *) redisCommand(m_redisContext,"LRANGE %s:%s 0 -1", key.c_str(), "data");
			if(reply->type == REDIS_REPLY_ARRAY)
			{
				data = new double[reply->elements];

				for(int i = 0; i < reply->elements; i++)
				{
					double value = atof(reply->element[i]->str);
					data[i] = value;
				}
			}
			else{

				//TODO: error handling
				throw 0;
			}
			freeReplyObject(reply);

			//create corresponding xMatrix
			xMatrix<double> matrix(data, dimensions, memPermission::owner);
			return matrix;

		} else {
			throw 0;
			//TODO: error handling
		}
	}
}
