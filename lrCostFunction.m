function [J , grad] = lrCostFunction (theta , X ,y , lambda) 
  m = length(y); 
  z = X*theta; 
  h = sigmoid(z); 
  theta(1,1) = 0; 
  %Regularized Cost Function 
  J = (1/m)* ((-y')*log(h) - (1-y)'*log(1-h)) + lambda/(2*m)*(theta' * theta); 
  grad = (1/m)* X'*(h-y) + (lambda/m)*theta; 
  grad(1,1) = (1/m)* X'(1,:) * (h-y); 
  grad = grad(:); 
 
end  