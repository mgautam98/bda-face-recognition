function [features] = FE2
    no_classes = 10;
    features = [];

    for i = 1:no_classes
        base = strcat('.\images\class', strcat(string(i), '\'));
        directory = strcat(base, '\*.jpg');
        srcFiles = dir(directory);
        features_t = [];
        
        for j = 1:length(srcFiles)
            filename = strcat(base, srcFiles(j).name);
            img = imread(filename);
            img = rgb2gray(img);
            gaborArray = gaborFilterBank(5, 4, 39, 39);
            features_temp = gaborFeatures(img, gaborArray, 512, 512);
            features_ab1 = (features_temp)';
            features_t(j, :) = [features_ab1, i];
        end

        features = vertcat(features, features_t);
    end

end
