OPTION = -g -std=c++0x

phony: example

xWaveletTransform.o: xWaveletTransform.cpp
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp xWaveletTransform.o
	clang++ $(OPTION) -o $@ $+
