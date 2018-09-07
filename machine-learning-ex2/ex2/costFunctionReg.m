function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
sigmoid_vector = sigmoid(X*theta);
theta_item = theta(2:size(theta,1),:);
regularization_item = theta_item'*theta_item*lambda/(2*m);
J = (1/m)*sum(-y.*(log(sigmoid_vector))-(1-y).*(log(1-sigmoid_vector)))+regularization_item; 

grad_theta = lambda/m*[0;theta(2:size(theta,1))];
grad = grad_theta + X'*(sigmoid_vector-y)/m;


% =============================================================

end
