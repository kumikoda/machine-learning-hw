function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% Evaluate input X using vector multiplication
H = X*theta;

% Calculate Square difference 
D = (H-y) .^ 2;

% Sum across all samples 
S = sum(D);

% Complete the cost function
J = S / (2*m);


% =========================================================================

end
