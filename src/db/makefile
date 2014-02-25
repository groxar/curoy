OPTION = -g -std=c++0x
LIBOPTION = -lhiredis -I/usr/local/include/hiredis

phony: xMatrixRedisBinaryAdapter.o xMatrixRedisStringAdapter.o example

xMatrixRedisStringAdapter.o: xMatrixRedisStringAdapter.cpp
	clang++ $(OPTION) $(LIBOPTION) -o $@ -c $+

xMatrixRedisBinaryAdapter.o: xMatrixRedisBinaryAdapter.cpp
	clang++ $(OPTION) $(LIBOPTION) -o $@ -c $+

example: example.cpp xMatrixRedisStringAdapter.o xMatrixRedisBinaryAdapter.o
	clang++ $(OPTION) $(LIBOPTION) -o $@ $+
