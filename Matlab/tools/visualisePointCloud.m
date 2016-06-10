% %% load the point cloud (.las) using lasdata
% tunnel1 = lasdata ('/Users/januaditya/Thesis/exjobb-data/volvo/pc/47.las');
% tunnel2 = lasdata ('/Users/januaditya/Thesis/exjobb-data/volvo/pc/48.las');
% % 
% % %% extract the XYZ
% xyz1 = tunnel1.get_xyz;
% xyz2 = tunnel2.get_xyz;
% % 
% % %% build the matlab specific pointcloud
% pc1 = pointCloud(xyz1);
% pc2 = pointCloud(xyz2);

pcshow(pc1); hold on;
pcshow(pc2);