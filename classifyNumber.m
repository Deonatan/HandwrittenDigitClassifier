input_layer_size = 400; 
num_labels = 10; 
hidden_layer_size=25; 

%Loading and Displaying Data
fprintf('Loading and Randomly Visualize 100 Data ... \n') 
load('Data\HandWrittenSample.mat'); 
load('Data\DataWeights.mat'); 
m = size(X , 1) ;

rand_indices = randperm(m); 
select = X(rand_indices(1:100) , :); 

displayData(select); 
fprintf('Program paused. Press enter to continue.\n');
pause;

%Computing Cost Function and Gradient 
fprintf('\nTesting Logistic Regresssion Cost Function with regularization') 
theta_t = [-2 ; -1; 1;2]; 
X_t = [ones(5,1) reshape(1:15 , 5 ,3)/10]; 
y_t = ([1;0;1;0;1] >= 0.5); 
lambda_t = 3; 
[J grad] = lrCostFunction(theta_t , X_t , y_t,lambda_t); 

fprintf('\nCost : %f', J) 
fprintf('\nGradients: %f ,%f , %f, %f', grad) 

%One-vs-All Training 
fprintf('\nTraining One-vs-All Logistic Regression...\n')
lambda = 0.1; 
[all_theta] = oneVsAll(X,y,num_labels,lambda); 

%Predict for One-vs-All 
pred = predictOneVsAll(all_theta, X); 
fprintf('\nTraining Set Accuracy Using One vs All Classification : %f\n' ,...
           mean(double(pred == y))*100) 

fprintf('Program paused. Press enter to continue.\n');
pause;

%Predict by Neural Network 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy Using Neural Network: %f\n', ...
           mean(double(pred == y)) * 100);
           
rp = randperm(m);
for i = 1:m 
  fprintf('\n Displaying Example Image\n'); 
  displayData(X(rp(i), :)); 
  
  pred = predict(Theta1, Theta2 , X(rp(i) ,:)); 
  fprintf('\nNeural Network Prediction : %d (digit %d)\n' , pred, mod(pred, 10));
  
   s = input('Paused - press enter to continue, q to quit:','s');
    if s == 'q'
      break
    endif
endfor


  

