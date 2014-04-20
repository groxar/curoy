NVCC = nvcc
CUDACNFG= -arch=sm_20 -O2 -lcurand
OPTION = -g -std=c++0x
LIBOPTION = -lhiredis -L/usr/local/lib -I/usr/local/include/hiredis

phony: cuMatrixRedisBinaryAdapter.o

xMatrixRedisStringAdapter.o: xMatrixRedisStringAdapter.cpp
	clang++ $(OPTION) $(LIBOPTION) -o $@ -c $+

xMatrixRedisBinaryAdapter.o: xMatrixRedisBinaryAdapter.cpp
	clang++ $(OPTION) $(LIBOPTION) -o $@ -c $+

xMatrixFileAdapter.o: xMatrixFileAdapter.cpp
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp xMatrixRedisStringAdapter.o xMatrixRedisBinaryAdapter.o xMatrixFileAdapter.o
	clang++ $(OPTION) $(LIBOPTION) -o $@ $+

cuMatrixRedisBinaryAdapter.o: cuMatrixRedisBinaryAdapter.cpp 
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION) $(LIBOPTION)' -o $@ -c $+ 
