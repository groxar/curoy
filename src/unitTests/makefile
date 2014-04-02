CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_20 -O2 -lcurand
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -O3
DBLIBOPTION = -lhiredis -L/usr/local/lib -I/usr/local/include/hiredis

phony: cuMatrixUnit xMatrixUnit run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+


../lib/cudaMatrix.o:
	make -C ../lib

../ml/ann.o:
	make -C ../ml

cuMatrixUnit: cuMatrixUnit.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

annUnit: annUnit.cpp ../lib/cudaMatrix.o ../ml/ann.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nANNUNIT"
	./annUnit

powerANN: powerANN.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nANN"
	./powerANN

gradientUnit: gradientUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nGradient"
	./gradientUnit

featureNormalizeUnit: featureNormalizeUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nFeatureNormalize"
	./featureNormalizeUnit

dbUnitTest: dbUnit.cpp ../db/xMatrixFileAdapter.o ../db/xMatrixRedisBinaryAdapter.o
	$(CXX) $(OPTION) $(DBLIBOPTION) -o $@ $+

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	@echo "\n\nCUMATRIX"
	./cuMatrixUnit 
	@echo "\n\nXMATRIX"
	./xMatrixUnit
