%% basic projection

%% point
point = [3; 3];
translation = [2; 2];
theta = 90;

rotationMatrix = [cosd(theta) -sind(theta);
                  sind(theta)  cosd(theta)];

%% new translation
newPointTransl = translation + point;
newPointRotate = rotationMatrix * point;
newPointFinale = rotationMatrix * point + translation;

%% now, reverting back
% given rotation and translation matrix
originalPoint = rotationMatrix.' * (newPointFinale - translation);

%% now, known world position of camera
cameraPoint = [143425.6844188932 6394339.576775855 35.996698161994570];

%% extrinsic and intrinsic camera matrices
% R matrix
rvec = [0.9374521391997079, -0.3434782435455127, -0.05662316593745959;
        -0.08260335620060169, -0.06147257392411345, -0.994684778308958;
        0.3381718087811107, 0.9371466368002576, -0.08600005162601933];

% T matrix
tvec = [2061863.850036867;
        404959.7605953381;
        -6040933.256341981];

% C matrix
K = [1432, 0, 640;
     0, 1432, 480;
     0, 0, 1];

%% compute world point to pixel point
worldPoint = [143430.318, 6394363.127, 39.797];
a = rvec * worldPoint.' + tvec;
b = [rvec tvec; 0 0 0 1] * [worldPoint 1].';
c = [rvec tvec; 0 0 0 1]\[0 0 0 1].';
