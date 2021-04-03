% Notation: This fitness function is for demonstration 

function cost = jFitnessFunction(feat,label,X,HO)
  alpha   = 0.99; 
  beta    = 0.01;
  maxFeat = length(X); 
  if sum(X == 1) == 0
    cost = inf;
  else
    error    = jwrapperdnn(feat(:, X == 1),label,HO);
    num_feat = sum(X == 1);
    cost     = alpha * error + beta * (num_feat / maxFeat); 
  end
  end
  
  
  function error = jwrapperknn(sFeat,label,HO)
  %---// Parameter setting for k-value of KNN //
  k = 5;
  
  trainIdx = HO.training;        testIdx  = HO.test;
  xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
  xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);
  
  KNN   = fitcknn(xtrain,ytrain,'NumNeighbors',k);
  pred  = predict(KNN,xvalid); 
  Acc   = jAccuracy(pred,yvalid);
  
  [C2,~] = confusionmat(pred,yvalid);
  
  acc_testing=(C2(1,1)+ C2(2,2))/(C2(1,1)+C2(1,2)+C2(2,1)+C2(2,2));
  specificity= C2(2,2)/ (C2(2,2)+C2(2,1));
  sensitivity=C2(1,1)/(C2(1,1)+C2(1,2));
  
  % fprintf('\n Accuracy: %g %%',100 * Acc);
  % fprintf(' Cal Accuracy: %g %%',100 * acc_testing);
  % fprintf(' Specificity: %g %%',100 * specificity);
  % fprintf(' Sensitivity: %g %%',100 * sensitivity);
  error = 1 - Acc; 
  end
  
  function error = jwrapperrbe(sFeat, label, HO)
  
    trainIdx = HO.training;        testIdx  = HO.test;
    xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
    xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);
  
    spread = 1.3;
    T = ind2vec(ytrain');
    X = xtrain';
  
    net = newrbe(X,T,spread);
    Y = net(X);
  
    Y = net(xvalid');
    Yc = vec2ind(Y);
    Yc=Yc';
  
    [C2,~] = confusionmat(Yc,yvalid);
    acc_testing=(C2(1,1)+ C2(2,2))/(C2(1,1)+C2(1,2)+C2(2,1)+C2(2,2));
    specificity= C2(2,2)/ (C2(2,2)+C2(2,1));
    sensitivity=C2(1,1)/(C2(1,1)+C2(1,2));
  
    error = 1 - acc_testing;
  end
  
  function error = jwrapperpnn(sFeat, label, HO)
  
    trainIdx = HO.training;        testIdx  = HO.test;
    xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
    xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);
  
    spread = 1.3;
    T = ind2vec(ytrain');
    X = xtrain';
  
    net = newpnn(X,T);
    Y = sim(net, X);
  
    Y = sim(net, xvalid');
    Yc = vec2ind(Y);
    Yc=Yc';
    
    [C2,~] = confusionmat(Yc,yvalid);
    acc_testing=(C2(1,1)+ C2(2,2))/(C2(1,1)+C2(1,2)+C2(2,1)+C2(2,2));
    specificity= C2(2,2)/ (C2(2,2)+C2(2,1));
    sensitivity=C2(1,1)/(C2(1,1)+C2(1,2));
  
    error = 1 - acc_testing;
  end
  
  function error = jwrapperdnn(sFeat, label, HO)
  
    trainIdx = HO.training;        testIdx  = HO.test;
    xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
    xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);
  
    spread = 1.3;
    T = ind2vec(ytrain');
    X = xtrain';
  
    % Stacked DNN --------------------------------------------------------------------------------
    hiddenSize = 10;
    autoenc1 = trainAutoencoder(X,hiddenSize,...
        'L2WeightRegularization',0.001,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.05,...
        'DecoderTransferFunction','purelin');
    features1 = encode(autoenc1,X);
    hiddenSize = 10;
    autoenc2 = trainAutoencoder(features1,hiddenSize,...
        'L2WeightRegularization',0.001,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.05,...
        'DecoderTransferFunction','purelin',...
        'ScaleData',false);
    features2 = encode(autoenc2,features1);
    softnet = trainSoftmaxLayer(features2,T,'LossFunction','crossentropy');
  
    deepnet = stack(autoenc1,autoenc2,softnet);
  
    % ----------------------------------------------------------------------------------------------
  
    deepnet = train(deepnet,X,T);

    % Validation data
    Y = deepnet(xvalid');
    Yc = vec2ind(Y);
    Yc=Yc';
    
    [C2,~] = confusionmat(Yc,yvalid);
    acc_testing=(C2(1,1)+ C2(2,2))/(C2(1,1)+C2(1,2)+C2(2,1)+C2(2,2))
    specificity= C2(2,2)/ (C2(2,2)+C2(2,1));
    sensitivity=C2(1,1)/(C2(1,1)+C2(1,2));
  
    error = 1 - acc_testing;
  end
