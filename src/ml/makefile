CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = powerANN
OPTION = -std=c++0x -g

phony: $(TARGET) $(OBJDIR) run

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

$(TARGET): $(TARGET).cpp
	$(CXX) $(OPTION) $+ -o $@ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
