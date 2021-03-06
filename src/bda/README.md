# Binary Dragonfly Algorithm for Feature Selection

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/2e802d69-43b0-49cd-8121-792c643de940/74bfd0a2-6577-477c-a419-bb920010a910/images/1603353704.JPG)

## Introduction
* This toolbox offers Binary Dragonfly Algorithm ( BDA ) method
* The < Main.m file > illustrates the example of how BDA can solve the feature selection problem using benchmark data-set.


## Input
* *feat*     : feature vector ( Instances *x* Features )
* *label*    : label vector ( Instances *x* 1 )
* *N*        : number of dragonflies
* *max_Iter* : maximum number of iterations


## Output
* *sFeat*    : selected features
* *Sf*       : selected feature index
* *Nf*       : number of selected features
* *curve*    : convergence curve


### Example
```code
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho);

% Parameter setting
N        = 10; 
max_Iter = 100; 
% Perform feature selection 
[sFeat,Sf,Nf,curve] = jBDA(feat,label,N,max_Iter,HO);

% Accuracy 
Acc = jKNN(sFeat,label,HO);

% Plot convergence curve
plot(1:max_Iter,curve);
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('BDA'); grid on;
```
% [C2,~] = confusionmat(pred,yvalid);

% acc_testing=(C2(1,1)+ C2(2,2))/(C2(1,1)+C2(1,2)+C2(2,1)+C2(2,2));
% specificity= C2(2,2)/ (C2(2,2)+C2(2,1));
% sensitivity=C2(1,1)/(C2(1,1)+C2(1,2));

% fprintf(' Cal Accuracy: %g %%',acc_testing);
% fprintf(' Specificity: %g %%',specificity);
% fprintf(' Sensitivity: %g %%',sensitivity);

https://github.com/JiaqiHe/Detection_of_Surge_Arrestor/blob/db49017b67bc5a4acc533955fcabbbc4e8a7ac5d/Auto_Encoder/train_and_test.m

https://github.com/harrysimply/pulse_recognition/blob/f5c070e000b6a582398f82a12c439e2dafede565/divide_train_test_data.m

## Requirement
* MATLAB 2014 or above
* Statistics and Machine Learning Toolbox


