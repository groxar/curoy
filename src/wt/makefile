OPTION = -g -std=c++0x

phony: example

xFilter.o: xFilter.cpp
	clang++ $(OPTION) -o $@ -c $+

xWaveletTransform.o: xWaveletTransform.cpp xFilter.o
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp xWaveletTransform.o xFilter.o
	clang++ $(OPTION) -o $@ $+
