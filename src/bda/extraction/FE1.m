function [features] = FE1
    no_classes = 3;
    features = [];

    for j = 1:no_classes
        base = strcat('.\images\class', strcat(string(i), '\'));
        directory = strcat(base, '\*.jpg');
        srcFiles = dir(directory);
        features_t = [];

        for i = 1:length(srcFiles)
            filename = strcat(base, srcFiles(i).name);
            img = imread(filename);
            I = rgb2gray(img);
            GLCM2 = graycomatrix(I);
            feature = GLCMFeatures(GLCM2);
            A = feature;
            ds = struct2dataset(A);
            s(i, 1) = ([ds.autoCorrelation]);
            s(i, 2) = ([ds.clusterProminence]);
            s(i, 3) = ([ds.clusterShade]);
            s(i, 4) = ([ds.contrast]);
            s(i, 5) = ([ds.correlation]);
            s(i, 6) = ([ds.differenceEntropy]);
            s(i, 7) = ([ds.differenceVariance]);
            s(i, 8) = ([ds.dissimilarity]);
            s(i, 9) = ([ds.energy]);
            s(i, 10) = ([ds.entropy]);
            s(i, 11) = ([ds. homogeneity]);
            s(i, 12) = ([ds.informationMeasureOfCorrelation1]);
            s(i, 13) = ([ds.informationMeasureOfCorrelation2]);
            s(i, 14) = ([ds.inverseDifference]);
            s(i, 15) = ([ds.maximumProbability]);
            s(i, 16) = ([ds.sumAverage]);
            s(i, 17) = ([ds.sumEntropy]);
            s(i, 18) = ([ds.sumOfSquaresVariance]);
            s(i, 19) = ([ds.sumVariance]);
            s(i, 20) = j;
            features_t(i, :) = s(i, :);
        end

        features = vertcat(features, features_t);
    end

end
