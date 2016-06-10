function [  poses, ...
            matrices, ...
            correspondenceReshape, ...
            correspondenceRefinedReshape, ...
            numrow, ...
            numRrow] = readLog (startidx, numfiles, pathname)

%% read each frames' correspondences (numfilesx9)
poseFile = sprintf('%s/logPoses.txt', pathname);
poses = csvread(poseFile, 0, 0, [0 0 numfiles-1 8]);

%% read each frames' camera matrices (numfilesx24)
matFile = sprintf('%s/logMatrix.txt', pathname);
matrices = csvread(matFile, 0, 0, [0 0 numfiles-1 23]);

%% read each frames' 3D-2D correspondences
% prealloc, read the first first
correFile = sprintf('%s/%d-poseFile.txt', pathname, startidx);
correFileRefined = sprintf('%s/%d-poseRefined.txt', pathname, startidx);
correspondence = csvread(correFile);
correspondenceRefined = csvread(correFileRefined);
correspondenceConcat = correspondence;
correspondenceRefinedConcat = correspondenceRefined;
numrow     = zeros (numfiles);
numRrow    = zeros (numfiles);

numrow(1) = size(correspondence, 1);
numRrow(1) = size(correspondenceRefined, 1);

for k = 1:numfiles
% set the filename
    correFile = sprintf('%s/%d-poseFile.txt', pathname, startidx+k);
    correFileRefined = sprintf('%s/%d-poseRefined.txt', pathname, startidx+k);
% read as csv files
    correspondence = csvread(correFile); 
    correspondenceRefined = csvread(correFileRefined);
% save length of each row to a temp var
    numrow(k+1) = size(correspondence, 1);
    numRrow(k+1) = size(correspondenceRefined, 1);
    
% use the length to alloc a concatenated bin
    correspondenceConcat = [correspondenceConcat; correspondence]; %#ok<AGROW>
    correspondenceRefinedConcat = [correspondenceRefinedConcat; correspondenceRefined]; %#ok<AGROW>
end

% reshape the original
maxrowscorrespondence = max (numrow);
maxsizecorrespondence = zeros(maxrowscorrespondence(1), size(correspondence,2));
correspondenceReshape = zeros(numfiles, size(maxsizecorrespondence,1), size(maxsizecorrespondence,2));

for i=1:numfiles
    for j=1:numrow(i)
        correspondenceReshape(i,j,:) = correspondenceConcat(j,:);
    end
end

% reshape the refined data
maxrowsRcorrespondence = max (numRrow);
maxsizeRcorrespondence = zeros(maxrowsRcorrespondence(1), size(correspondenceRefined,2));
correspondenceRefinedReshape = zeros(numfiles, size(maxsizeRcorrespondence,1), size(maxsizeRcorrespondence,2));

for i=1:numfiles
    for j=1:numRrow(i)
        correspondenceRefinedReshape(i,j,:) = correspondenceRefinedConcat(j,:);
    end
end