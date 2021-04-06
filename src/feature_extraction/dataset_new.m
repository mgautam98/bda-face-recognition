dataset1 = FE2;
dataset2 = dataset1(:, 1:size(dataset1, 2) - 1);
dataset3 = DCT;
dataset4 = dataset3(:, 1:size(dataset3, 2) - 1);
dataset5 = SURF;

dataset = [dataset2, dataset4, dataset5];

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
