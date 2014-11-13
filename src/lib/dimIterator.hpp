#pragma once
#include <vector>

namespace curoy {
using namespace std;

inline size_t dimCompareIgnore1(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs){
	std::vector<size_t> templ;
	std::vector<size_t> tempr;
	size_t result = 0;

	for(auto it: lhs){
		if(it!=1)
			templ.push_back(it);
	}

	for(auto it: rhs){
		if(it!=1)
			tempr.push_back(it);
	}

	size_t end = std::min(templ.size(),tempr.size());
	result += std::max(templ.size(),tempr.size()) - end;
	for(size_t i = 0; i < end; ++i){
		if(templ[i] != tempr[i])
			++result;
	}

	return result;
}
inline size_t dimCompare(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs){
	size_t result = 0;

	size_t end = std::min(lhs.size(),rhs.size());
	result += std::max(lhs.size(),rhs.size()) - end;
	for(size_t i = 0; i < end; ++i){
		if(lhs[i] != rhs[i])
			++result;
	}

	return result;
}
}
