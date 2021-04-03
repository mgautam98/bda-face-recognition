%-------------------------------------------------------------------%
%  Binary Dragonfly Algorithm (BDA) demo version                    %
%-------------------------------------------------------------------%

%---Inputs-----------------------------------------------------------
% feat     : feature vector (instances x features)
% label    : label vector (instances x 1)
% N        : Number of dragonflies
% max_Iter : Maximum number of iterations

%---Outputs-----------------------------------------------------------
% sFeat    : Selected features
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%---------------------------------------------------------------------

%% Binary Dragonfly Algorithm
clc; clear; close
% Benchmark data set
noFolds = 10;
dataset = xlsread('tempfile.xlsx');

[m, n] = size(dataset);
classIndex = n;
totalNoFeatures = n - 1;
Data = dataset(:, 1:totalNoFeatures);
Label = dataset(:, classIndex);

% Set 20% data as validation set
ho = 0.2;
% Hold-out method
HO = cvpartition(Label, 'HoldOut', ho, 'Stratify', false);

% Parameter setting
N = 10;
max_Iter = 5;
% Perform feature selection
[sFeat, Sf, Nf, curve] = jBDA(Data, Label, N, max_Iter, HO);

% Accuracy
Acc = jKNN(sFeat, Label, HO);

% Plot convergence curve
plot(1:max_Iter, curve);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('BDA'); grid on;
