function [features] = FE
    no_classes = 3;
    
    features = [];

    for i = 1:no_classes
        base = strcat('.\images\class', strcat(string(i), '\'));
        directory = strcat(base, '\*.jpg');
        srcFiles = dir(directory);
        features_t = [];

        for j = 1:length(srcFiles)
            filename = strcat(base, srcFiles(i).name);
            I = imread(filename);
            mask = ones(size(I(:, :, 1)));
            quantize = 16;
            [SRE, LRE, GLN, RP, RLN, LGRE, HGRE] = glrlm(I, quantize, mask);
            features_t(j, :) = [SRE, LRE, GLN, RP, RLN, LGRE, HGRE, i];
        end

        features = vertcat(features, features_t);
    end

    name = {'SRE' 'LRE' 'GLN' 'RP' 'RLN' 'LGRE' 'HGRE' 'CLASSLABEL'};

    fearures_final = [name; num2cell(features)];
end
