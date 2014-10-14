CC = gcc
CXX = g++
NVCC = nvcc
CUDACNFG= -arch=sm_20 -ccbin=/usr/bin/g++
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g

phony: cuMatrix.o cudaMatrix.o 

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

xMatrix.o: xMatrix.hpp
	$(CXX) $(OPTION) -o $@ -c $+

cudaMatrix.o: cuMatrix.cu
	$(NVCC) $(CUDACNFG) -o $@ -Xcompiler '-fPIC' --shared $+

cuMatrix.o: cuMatrix.hpp 
	$(CXX) $(OPTION) -o $@ -c -x c++ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
