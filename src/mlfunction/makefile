CC = gcc
CXX = g++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = powerANN

phony: $(TARGET) $(OBJDIR) run

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

$(TARGET): $(TARGET).cu
	$(NVCC) $(CUDACNFG) -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(TARGET)
