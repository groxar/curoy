#pragma once
#include "../lib/cuMatrix.hpp"

using namespace std;

namespace curoy{
	public class ann{
		double predict(const cuMatrix<double>& theta, const cuMatrix<double> xInput){
			m = size(X, 1);
			num_labels = size(Theta2, 1);

			% You need to return the following variables correctly 
			p = zeros(size(X, 1), 1);

			% ====================== YOUR CODE HERE ======================
			% Instructions: Complete the following code to make predictions using
			%               your learned neural network. You should set p to a 
			%               vector containing labels between 1 to num_labels.
			%
			% Hint: The max function might come in useful. In particular, the max
			%       function can also return the index of the max element, for more
			%       information see 'help max'. If your examples are in rows, then, you
			%       can use max(A, [], 2) to obtain the max for each row.
			%

			X = [ones(m, 1) X];
			a = sigmoid(X*Theta1');
			m = size(a,1);
			a = [ones(m, 1) a];
			a = a*Theta2';
			a = sigmoid(a);

			[val, p]=max(a,[],2);
		}
	};
}
