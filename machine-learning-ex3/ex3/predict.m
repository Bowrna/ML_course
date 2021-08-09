function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
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

% Layer 1 bias
a1 = [ones(m, 1) X];
fprintf('A1 size %f:',size(a1))
fprintf('Theta1 size %f:',size(Theta1))
fprintf('Theta2 size %f:',size(Theta2))
z1 = a1 * Theta1';
h1 = sigmoid(z1);
a2 = [ones(m,1) h1];
fprintf('A2 size %f:',size(a2))
z2 = a2 * Theta2';
h = sigmoid(z2);
[~, p] = max(h, [], 2); 







% =========================================================================


end
