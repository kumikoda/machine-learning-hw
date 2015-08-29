function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% http://www.holehouse.org/mlclass/07_Regularization.html 

% this is from previous exercise, which is already vectorized
% hx = sigmoid(X * theta);
% costs = -y' * log(hx) - (1 - y')*log(1 - hx);
% J = sum(costs)/m;
% grad = X' * (hx - y) / m;

% now we just need to add the regularization part 
hx = sigmoid(X * theta);
costs = -y' * log(hx) - (1 - y')*log(1 - hx);
penalty = sum(lambda * theta(2:end).^2)/(2*m);
J =  sum(costs)/m + penalty;

% like in the hint above, we dont want to add a penalty for j=0
tmp = theta;
tmp(1) = 0;

% partially stolen from 
% http://stackoverflow.com/questions/19824293/regularized-logistic-regression-code-in-matlab
grad_pen = + lambda .* tmp ./ m;
grad =((hx - y)' * X / m)' + grad_pen; 

% seriously though, the answer was in the instructions itself... 
% please read the entire question prompt before googling :)

% =============================================================

grad = grad(:);

end
