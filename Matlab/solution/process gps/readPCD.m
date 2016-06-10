function [ cloud ] = readPCD (filename)

%% open the .PCD
fileID = fopen(filename,'r');

%% skip the header
for k=1:11
    fgets(fileID);
end

%% set the format
formatSpec = '%f %f %f';
sizeA = [3 Inf];

cloud = fscanf(fileID,formatSpec,sizeA);
cloud = cloud.';

fclose(fileID);