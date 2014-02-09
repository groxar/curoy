CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
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
	$(NVCC) $(CUDACNFG) -o $@ -c $+

cuMatrix.o: cuMatrix.hpp 
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ -c -x c++ $+ 


clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
