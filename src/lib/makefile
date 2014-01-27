CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
XMATRIX = xMatrix
CUMATRIX = cuMatrix
OPTION = -std=c++0x -g

phony: $(CUMATRIX).o cudaMatrix.o

$(XMATRIX).o: $(XMATRIX).hpp
	$(CXX) $(OPTION) -o $@ -c $+

cudaMatrix.o: cuMatrix.cu
	$(NVCC) $(CUDACNFG) -o $@ -c $+

$(CUMATRIX).o: $(CUMATRIX).hpp 
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ -c -x c++ $+ 

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
