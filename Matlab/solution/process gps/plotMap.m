% load first
load('160222_124022_LocalProcessed.mat');
load('plot_thesis_output.mat');

% point cloud
cloudpath = '/Users/januaditya/Workspace/_matlab/exjobb/solution/plotting log/cloud/gnistangtunneln-semifull-voxelized.pcd';
cloudxyz = readPCD (cloudpath);
pcloud = pointCloud (cloudxyz);
gridStep = 3;
ptCloudA = pcdownsample(pcloud,'gridAverage',gridStep);

figure(2);

subplot(2,2,1);
plot (rtStruct.SwerefEasting, rtStruct.SwerefNorthing); hold on;
plot (poseOutput(:,1), poseOutput(:,2), 'x');

subplot(2,2,2);
plot (rtStruct.SwerefEasting, rtStruct.SwerefNorthing); hold on;
plot (poseOutput(:,1), poseOutput(:,2), 'x');

subplot(2,2,3);
plot (rtStruct.SwerefEasting, rtStruct.SwerefNorthing); hold on;
plot (ptCloudA.Location(:,1), ptCloudA.Location(:,2), '.');

subplot(2,2,4);
plot (rtStruct.SwerefEasting, rtStruct.SwerefNorthing); hold on;
plot (poseOutput(:,1), poseOutput(:,2), 'x');
plot (ptCloudA.Location(:,1), ptCloudA.Location(:,2), '.');


% subplot(2,2,4);
% plot (rtStruct.SwerefEasting(805450:806000), rtStruct.PosAlt(805450:806000)); hold on;
% plot (poseOutput(:,1), poseOutput(:,3), 'x'); 

% lat = [48.8708 51.5188 41.9260 40.4312 52.523 37.982]; 
% lon = [2.4131 -0.1300 12.4951 -3.6788 13.415 23.715]; 
% plot(lon,lat,'.r','MarkerSize',20);
% plot_google_map;