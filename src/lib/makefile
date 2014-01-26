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

$(XMATRIX).o: $(XMATRIX).cpp
	$(CXX) -c $(OPTION) -o $@ $+

$(CUMATRIX).o: $(CUMATRIX).cpp
	$(NVCC) -c $+ -Xcompiler '$(OPTION)' -o $@ 

$(TARGET): $(TARGET).cpp $(CUMATRIX).o
	$(NVCC) $+ -Xcompiler '$(OPTION)' -o $@ 
$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
