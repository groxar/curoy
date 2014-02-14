#pragma once
#include <vector>

using namespace std;
size_t dimCompare(const vector<size_t>& lhs, const vector<size_t>& rhs){
	vector<size_t> templ;
	vector<size_t> tempr;
	size_t result = 0;

	for(auto it: lhs){
		if(it!=1)
			templ.push_back(it);
	}

	for(auto it: rhs){
		if(it!=1)
			tempr.push_back(it);
	}

	size_t end = min(templ.size(),tempr.size());
	result += max(templ.size(),tempr.size()) - end;
	for(size_t i = 0; i < end; ++i){
		if(templ[i] != tempr[i])
			++result;
	}

	return result;
}
