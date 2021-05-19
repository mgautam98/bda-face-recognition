function [features] = SURF
    no_classes = 10;
    features = [];

    for i = 1:no_classes
        base = strcat('.\images\class', strcat(string(i), '\'));
        directory = strcat(base, '\*.jpg');
        srcFiles = dir(directory);
        features_t = [];

        for j = 1:length(srcFiles)
            filename = strcat(base, srcFiles(j).name);
            I = imread(filename);
            I = rgb2gray(I);
            points = detectSURFFeatures(I);
            features_t(j, :) = [points.selectStrongest(10).Metric; i];
        end

        features = vertcat(features, features_t);
    end

end
