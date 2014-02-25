OPTION = -g -std=c++0x
LIBOPTION = -lhiredis -I/usr/local/include/hiredis

phony: xMatrixRedisStringAdapter.o example

xMatrixRedisStringAdapter.o: xMatrixRedisStringAdapter.cpp
	clang++ $(OPTION) $(LIBOPTION) -o $@ -c $+

example: example.cpp xMatrixRedisStringAdapter.o
	clang++ $(OPTION) $(LIBOPTION) -o $@ $+
