OPTION = -g -std=c++0x

phony: example

Filter.o: Filter.cpp
	clang++ $(OPTION) -o $@ -c $+

SymmetricPadding.o: SymmetricPadding.cpp
	clang++ $(OPTION) -o $@ -c $+

PeriodicPadding.o: PeriodicPadding.cpp
	clang++ $(OPTION) -o $@ -c $+

ZeroPadding.o: ZeroPadding.cpp
	clang++ $(OPTION) -o $@ -c $+

ConstantPadding.o: ConstantPadding.cpp
	clang++ $(OPTION) -o $@ -c $+

WaveletTransformator.o: WaveletTransformator.cpp Filter.o SymmetricPadding.o
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp WaveletTransformator.o Filter.o SymmetricPadding.o
	clang++ $(OPTION) -o $@ $+

benchmark: benchmark.cpp WaveletTransformator.o Filter.o SymmetricPadding.o
	clang++ $(OPTION) -o $@ $+