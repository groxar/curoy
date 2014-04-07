OPTION = -g -std=c++0x
NVCC = nvcc
CUDACNFG= -arch=sm_20

phony: benchmark

cubench.o: bench.cu
	$(NVCC) $(CUDACNFG) -o $@ -c $+

xbench.o: bench.cpp
	clang++ $(OPTION) -o $@ -c $+

benchmark: benchmark.cpp xbench.o cubench.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+
