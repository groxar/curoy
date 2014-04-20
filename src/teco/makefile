CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_20 -O2 -lcurand
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -O3
DBLIBOPTION = -lhiredis -L/usr/local/lib -I/usr/local/include/hiredis

phony: clustering

../lib/cudaMatrix.o:
	make -C ../lib

../ml/ann.o:
	make -C ../ml

../db/cuMatrixRedisBinaryAdapter.o:
	make -C ../db


clustering: clustering.cpp ../lib/cudaMatrix.o ../db/cuMatrixRedisBinaryAdapter.o ../ml/ann.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION) $(DBLIBOPTION)' -o $@ $+ 
	@echo "\n\nClustering"
	./clustering

clean:
	rm xMatrixUnit cuMatrixUnit

