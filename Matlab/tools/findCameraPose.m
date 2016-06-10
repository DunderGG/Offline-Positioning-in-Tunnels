%% load the mat
load ('camera_matrix.mat');

%% get the camera matrices
% 3d points
queryPoint = xyz1;

% Rotation matrix
rotationMatrix = R_33;

% Translation matrix
translationMatrix = t;

%% get 3x3 the intrinsic camera parameter
intrinsicMatrix = C;

%% get 3x4 the extrinsic camera parameter
extrinsicMatrix = [rotationMatrix, t; 0 0 0 1];

%% get 3x4 camera projection matrix
I_C = [eye(3,3), transpose([0,0,0])];

%% project 3d query point into 3d with respect to camera coordinate
% still in camera 3d coordinate
u_v = intrinsicMatrix * I_C * extrinsicMatrix * queryPoint;

% normalize it in 2d pixels
u_v = [u_v(1)/u_v(3); u_v(2)/u_v(3); 1];

%% to get the camera pose
cameraCoord  = -1 * transpose(rotationMatrix)*t;
