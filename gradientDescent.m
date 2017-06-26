function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for i = 1:num_iters
	temp0 = X * theta;
	
	temp0 = temp0 .- y;
	temp1 = temp0 .* X(:,2:2);
	temp1 = sum(temp1);
	temp0 = sum(temp0);
	betaa1 = alpha * temp1;
	betaa1 = betaa1 / m;
	betaa = alpha * temp0;
	betaa = betaa / m;
	theta0 = theta(1) - betaa;
	theta1 = theta(2) - betaa1;
	theta(1) = theta0;
	theta(2) = theta1;
end	

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
   

end
