CC = gcc
CXX = g++
NVCC = nvcc
CUDACNFG= -arch=sm_20 -lcurand -ccbin=/usr/bin/g++
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
	./cuMatrixUnit

annUnit: annUnit.cpp ../lib/cudaMatrix.o ../ml/ann.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+
	@echo "\n\nANNUNIT"
	./annUnit

annUnit2: annUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -l../ml/libann.so -Xcompiler '$(OPTION)' -o $@ $+
	@echo "\n\nANNUNIT"
	./annUnit

kmeansUnit: kmeansUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+
	@echo "\n\nKMEANSUNIT"
	time ./kmeansUnit

svmUnit: svmUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+
	@echo "\n\nSVMUNIT"
	./svmUnit

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

wtUnitTest: wtUnitTest.cpp ../wt/WaveletTransformator.o ../wt/Filter.o ../wt/SymmetricPadding.o
	$(CXX) $(OPTION) $(DBLIBOPTION) -o $@ $+

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	@echo "\n\nCUMATRIX"
	./cuMatrixUnit
	@echo "\n\nXMATRIX"
	./xMatrixUnit
