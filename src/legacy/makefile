CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = cuMatrix
XMATRIX = xMatrix
XMATRIXT = xMatrixTemplateVariadic
CUMATRIX = cuMatrix
OPTION = -std=c++0x -g

phony: $(XMATRIX) $(TARGET) 

$(XMATRIX): $(XMATRIX).cpp
	$(CXX) $(OPTION) -o $@ $+

$(XMATRIXT): $(XMATRIXT).cpp
	$(CXX) $(OPTION) -o $@ $+

$(CUMATRIX): $(CUMATRIX).cpp
	$(NVCC) $+ -Xcompiler '$(OPTION)' -o $@ 

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
