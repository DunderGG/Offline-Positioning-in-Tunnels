%% load the mat
load ('entranceTunnelSmall.mat');

%% create delaunayTriangulation class
DT = delaunayTriangulation(entranceXYZ);

%% get the faces
entranceConnectivityList = DT.ConnectivityList;

%% plot
% faceColor  = [0.6875 0.8750 0.8984];
% figure
% tetramesh(DT,'FaceColor',faceColor,'FaceAlpha',0.3);