CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = cudaMatrixTest
XMATRIX = xMatrix
CUMATRIX = cuMatrix
OPTION = -std=c++0x -g

phony: $(TARGET) run

$(XMATRIX).o: $(XMATRIX).hpp
	$(CXX) $(OPTION) -o $@ -c $+

cudaMatrix.o: cuMatrix.cu
	$(NVCC) $(CUDACNFG) -o $@ -c $+

$(CUMATRIX).o: $(CUMATRIX).hpp 
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ -c -x c++ $+ 

$(TARGET): $(TARGET).cpp $(CUMATRIX).o cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
