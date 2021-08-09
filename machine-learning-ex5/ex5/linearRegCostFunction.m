function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%fprintf('Size of X:%f',size(X));% (12 * 2)
%fprintf('Size of y:%f',size(y));% (12 * 1)
%fprintf('Size of theta:%f',size(theta));% (2 * 1) 
h_theta = X * theta;% (12 * 1)
J = 1 / (2*m) * ( h_theta - y)' * (h_theta - y) + ...
    (lambda / (2*m)) * (theta(2:end))' * (theta(2:end));
thetazero = theta;% (2 * 1)
thetazero(1) = 0;% (2 * 1)
grad = (1 / m) * (h_theta - y)' * X + ...
      (lambda / m) * thetazero';
grad = grad(:);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
