function p = predict(Theta1, Theta2 , X) 
  m = size(X , 1) ; 
  num_labels = size(Theta2 , 1); 
  p = zeros(size(X ,1) , 1); 
  X = [ones(m,1) X] ; 
  z2 = X * Theta1' ; 
  a2 = sigmoid(z2) ; 
  s = size(a2 , 1); 
  a2 = [ones(m,1) a2];
  z3 = a2 * Theta2' ; 
  a3 = sigmoid(z3) ;
  [probability , p] = max(a3, [] , 2); 
endfunction 