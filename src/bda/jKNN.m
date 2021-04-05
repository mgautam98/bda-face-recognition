function Acc = jKNN(sFeat, label, HO)
    %---// Parameter setting for k-value of KNN //
    k = 5;

    trainIdx = HO.training; testIdx = HO.test;
    xtrain = sFeat(trainIdx, :); ytrain = label(trainIdx);
    xvalid = sFeat(testIdx, :); yvalid = label(testIdx);

    KNN = fitcknn(xtrain, ytrain, 'NumNeighbors', k);
    pred = predict(KNN, xvalid);
    % Acc = jAccuracy(pred, yvalid);

    [C2, ~] = confusionmat(pred, yvalid);

    Acc = (C2(1, 1) + C2(2, 2)) / (C2(1, 1) + C2(1, 2) + C2(2, 1) + C2(2, 2));
    specificity = C2(2, 2) / (C2(2, 2) + C2(2, 1));
    sensitivity = C2(1, 1) / (C2(1, 1) + C2(1, 2));

    fprintf('\n Accuracy: %g %%', 100 * Acc);
    fprintf(' Cal Accuracy: %g %%', 100 * acc_testing);
    fprintf(' Specificity: %g %%', 100 * specificity);
    fprintf(' Sensitivity: %g %%', 100 * sensitivity);
end
