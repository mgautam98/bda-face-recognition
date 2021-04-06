function [features] = SURF
    no_classes = 10;
    features = [];

    for i = 1:no_classes
        base = strcat('.\images\class', strcat(string(i), '\'));
        directory = strcat(base, '\*.jpg');
        srcFiles = dir(directory);
        features_t = [];

        for j = 1:length(srcFiles)
            filename = strcat(base, srcFiles(i).name);
            I = imread(filename);
            I = rgb2gray(I);
            D = dct2(I);
            S = D(1:5, 1:5);
            features_t(j, :) = [reshape(S, [1,25]), i];
        end

        features = vertcat(features, features_t);
    end

end
