%% load the point cloud (.las) using lasdata
gnistangtunneln = lasdata ('/Users/januaditya/Thesis/exjobb-data/volvo/pc/47.las');
% 
% %% extract the XYZ
gnistangtunnelnXYZ = gnistangtunneln.get_xyz;
% 
% %% build the matlab specific pointcloud
gnistangtunnelnPC = pointCloud(gnistangtunnelnXYZ);

%% export to a .ply file that can be load by MESH in openCV
queryPoint = [1.434381200000000e+05, 6394381, 32.617863000000000];
[entranceIndices, entranceDists] = findNearestNeighbors(gnistangtunnelnPC, queryPoint, 5000000);
entranceXYZ = gnistangtunnelnXYZ(entranceIndices, 1:3);

%% writeFile
ptCloud = pointCloud(entranceXYZ);
pcshow(ptCloud);
pcwrite (ptCloud, 'entranceTunnel', 'PLYFormat', 'ascii');