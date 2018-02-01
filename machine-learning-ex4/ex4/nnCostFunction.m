function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
	
yy = zeros(m,num_labels);
 for i = 1:num_labels
        yy(:, i) = y == i;
    end;

	hidden_layer_z = X * Theta1';%dimensions m*25
	sigmoid_1 = sigmoid(hidden_layer_z);
	
	a_hidden = sigmoid_1;%dimensions m*25
	sigmoid_1 = [ones(m,1) sigmoid_1];
	a_hidden_1 = sigmoid_1;
	ans = sigmoid_1 * Theta2';
	ans = sigmoid(ans);
	predicted_y = ans;
	ans_1 = log(ones(size(ans))-ans);
	ans = log(ans);
	y_1 = ones(size(yy))-yy;
	ans = yy.*ans;
	ans_1 = ans_1.*y_1;
	ans = ans + ans_1;
	J = sum(sum(ans),2);
	J = J/m;
	J = -1*J;
J = J + lambda * sum(sum(Theta1(:,2:input_layer_size+1).^2),2)/(2*m);
J = J + lambda * sum(sum(Theta2(:,2:hidden_layer_size+1).^2),2)/(2*m);
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
del_out = predicted_y - yy;
del_hl = (del_out * Theta2) .* (a_hidden_1 .* (ones(size(a_hidden_1)) - a_hidden_1)); 
%del_hl is of dimension m*26
a_one = X;
delta_1 = del_hl(:,2:end)' * a_one;
delta_2 = del_out' * a_hidden_1;
% -------------------------------------------------------------

% =========================================================================
Theta1_grad = (delta_1 + (lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)]))/m;
Theta2_grad = (delta_2 + (lambda * [zeros(size(Theta2,1),1), Theta2(:,2:end)]))/m;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
