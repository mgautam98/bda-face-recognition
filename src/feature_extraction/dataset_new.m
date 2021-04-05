% dataset1 = FE;
% dataset2 = dataset1(:, 1:size(dataset1, 2) - 1);
% dataset3 = FE1;
dataset4 = FE2;
dataset5 = dataset4(:, 1:size(dataset4, 2) - 1);
dataset6 = SURF;
dataset7 = dataset6(:, 1:size(dataset6, 2) - 1);
% dataset = [dataset2, dataset5, dataset7, dataset3];

dataset = [dataset5, dataset6];

rand_sequence = randperm(size(dataset, 1)); %find possible combination fom 1 to no of rows
temp_dataset = dataset; %  assign data set to temp

dataset = temp_dataset(rand_sequence, :); % arrange data set in random sequence order

for i = 1:size(dataset, 2) - 1 % from 1 to (number of column-1)[normalize the input]

    if max(dataset(:, i)) ~= min(dataset(:, i))
        dataset(:, i) = (dataset(:, i) - min(dataset(:, i))) / (max(dataset(:, i)) - min(dataset(:, i))) * 2 - 1;
    else
        dataset(:, i) = 0;
    end

end

filename = 'features.mat';
% xlswrite(filename, dataset, 'Sheet1');
label = dataset(:, size(dataset, 2));
feat = dataset(:, 1:size(dataset, 2)-1);

size(feat)
size(label)

save(filename, 'feat', 'label');
