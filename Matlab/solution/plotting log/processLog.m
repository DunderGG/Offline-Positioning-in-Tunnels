%% docked figure
% dock
% set(0,'DefaultFigureWindowStyle','docked');
% normal
set(0,'DefaultFigureWindowStyle','normal');

%% draw the point cloud
cloudpath = 'cloud/gnistangtunneln-semifull-voxelized.pcd';
cloudxyz = readPCD (cloudpath);
pcloud = pointCloud (cloudxyz);
gridStep = 0.2;
ptCloudA = pcdownsample(pcloud,'gridAverage',gridStep);

%% load the log

% start initiate the start index and frame length
startidx = 433;
framelength = 50;
logpath = 'log-may-26-long';

% read the log given the path
[pose, mat, corr, corrR, numOf2Dto3D, numOf2Dto3DRefined] = readLog (startidx, framelength, logpath);
poseOutput = pose((1:framelength), 4:6);

figure (2);
for i=1:1:framelength
    scatter3(corr(i,1:numOf2Dto3D(i),1), corr(i,1:numOf2Dto3D(i),2), corr(i,1:numOf2Dto3D(i),3), '.');
end

%% draw the vehicle's camera poses (as trajectory)
K = [1432, 0, 640; 0, 1432, 480; 0, 0, 1];
p_minrange      = 0;
p_maxrange      = 20;
cameraSize = 0.4;
figure(1);
% filename = 'tracking.gif';
pcshow (ptCloudA); hold on;
for frame = 1:1:size(numOf2Dto3D,1)
    % draw camera pose given a camera center and rmat3x3
    rvec9x1 = mat(frame, 1:9);
    rvec3x3 = reshape(rvec9x1, [3,3]).';
    tvec3x1 = mat(frame, 10:12);
    
    % draw search area as a square
    invertT = [ inv(rvec3x3), -inv(rvec3x3)*(tvec3x1.'); 0 0 0 1 ];
    c_lefttop = K \ [p_maxrange * 0   ; p_maxrange * 0   ; p_maxrange];
    c_leftbot = K \ [p_maxrange * 0   ; p_maxrange * 960 ; p_maxrange];
    c_rigtbot = K \ [p_maxrange * 1280; p_maxrange * 960 ; p_maxrange];
    c_rigttop = K \ [p_maxrange * 1280; p_maxrange * 0   ; p_maxrange];
    w_lefttop = invertT * [c_lefttop; 1];
    w_leftbot = invertT * [c_leftbot; 1];
    w_rigtbot = invertT * [c_rigtbot; 1];
    w_rigttop = invertT * [c_rigttop; 1];   
    line_centertolefttop = [pose(frame, 1:3); w_lefttop(1:3).'];
    line_centertoleftbot = [pose(frame, 1:3); w_leftbot(1:3).'];
    line_centertorigtbot = [pose(frame, 1:3); w_rigtbot(1:3).'];
    line_centertorigttop = [pose(frame, 1:3); w_rigttop(1:3).'];
    plot3(line_centertolefttop(1:2, 1), line_centertolefttop(1:2, 2), line_centertolefttop(1:2, 3)); 
    plot3(line_centertoleftbot(1:2, 1), line_centertoleftbot(1:2, 2), line_centertoleftbot(1:2, 3)); 
    plot3(line_centertorigttop(1:2, 1), line_centertorigttop(1:2, 2), line_centertorigttop(1:2, 3)); 
    plot3(line_centertorigtbot(1:2, 1), line_centertorigtbot(1:2, 2), line_centertorigtbot(1:2, 3)); 
%     scatter3(corr(frame,1:numOf2Dto3D(frame),1), corr(frame,1:numOf2Dto3D(frame),2), corr(frame,1:numOf2Dto3D(frame),3), 10, 'filled');
%     plot3(corr(frame,1:numOf2Dto3D(frame),1), corr(frame,1:numOf2Dto3D(frame),2), corr(frame,1:numOf2Dto3D(frame),3), 'x');
    drawPose (pose(frame, 4:6), rvec3x3, cameraSize);
    
    % create gif
%     frameGif = getframe(1);
%       im = frame2im(frameGif);
%       [imind,cm] = rgb2ind(im,256);
%       if frame == 1;
%           imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
%       else
%           imwrite(imind,cm,filename,'gif','WriteMode','append');
%       end
end


%% draw 2D x-y (lat-long) trajectories
figure;
subplot(1,2,1);
plot (pose(1:framelength, 1), pose(1:framelength,2));
xlim ([143420 143440]);
ylim auto;

%% draw 2D y-z (long-alt) trajectories
subplot(1,2,2);
plot (pose(1:framelength, 2), pose(1:framelength,3));
ylim ([0 50]);