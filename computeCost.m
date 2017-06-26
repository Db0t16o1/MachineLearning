function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

temp = X * theta; 
temp = temp.- y;
for i = 1:m
	J = J + temp(i)*temp(i);
	 %J = J + (theta(0) + theta(1)*X(i) - y(i))**2;
	end;
J = J/2;
J = J/m;

% =========================================================================

end
