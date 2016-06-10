%% 3d point
worldPoints =  [143430.318, 6394363.127, 39.797;
                143434.166, 6394361.662, 39.842;
                143432.108, 6394362.287, 39.617;
                143432.948, 6394364.927, 37.656;
                143427.658, 6394362.027, 38.376;
                143436.316, 6394359.472, 38.452;
                143427.048, 6394361.520, 33.577;
                143430.465, 6394361.396, 38.098;
                143437.223, 6394361.204, 39.037;
                143432.753, 6394362.541, 39.446];

%% R matrix
rvec = [0.9374521391997079, -0.3434782435455127, -0.05662316593745959;
        -0.08260335620060169, -0.06147257392411345, -0.994684778308958;
        0.3381718087811107, 0.9371466368002576, -0.08600005162601933];
    
%% T matrix
tvec = [2061863.850036867;
        404959.7605953381;
        -6040933.256341981];
    
%% 3d point transformed on camera space
cameracoordinate = rvec * worldPoints(1,:).' + tvec;

%% check whether we will get the 3d point again
reverseTvec = [inv(rvec), -inv(rvec)*tvec; 0, 0, 0, 1];
worldcoordinate = reverseTvec * [cameracoordinate; 1];

%% C matrix
K = [1432, 0, 640;
     0, 1432, 480;
     0, 0, 1];
 
%% pixel coordinate 
pixelcoordinate = K*cameracoordinate;
normalizedcoordinate = [pixelcoordinate(1)/pixelcoordinate(3);
                        pixelcoordinate(2)/pixelcoordinate(3);
                        pixelcoordinate(3)/pixelcoordinate(3)];
    
%% revert back
pixels =   [397.210571, 145.146866;
            650.494934, 129.172379;
            519.567688, 131.898239;
            531.834473, 267.480103;
            239.835358, 207.141220;
            834.740051, 174.580566;
            211.190155, 510.402740;
            437.319458, 218.244186;
            845.259948, 160.413910;
            559.729248, 170.678528];

sometemp = reverseTvec * [pixels(1,:) 1 1].';        
        
cameracenter = inv(K) * [0,0,0].';
cameraspace0  = inv(K) * [pixels(1,:) 1].';
cameraspace1  = inv(K) * [pixels(2,:) 1].';
cameraspace2  = inv(K) * [pixels(3,:) 1].';
cameraspace3  = inv(K) * [pixels(4,:) 1].';
cameraspace4  = inv(K) * [pixels(5,:) 1].';
cameraspace5  = inv(K) * [pixels(6,:) 1].';
cameraspace6  = inv(K) * [pixels(7,:) 1].';
cameraspace7  = inv(K) * [pixels(8,:) 1].';
cameraspace8  = inv(K) * [pixels(9,:) 1].';
cameraspace9  = inv(K) * [pixels(10,:) 1].';

corner0 = inv(K) * [0,0,1].';
corner1 = inv(K) * [1280,0,1].';
corner2 = inv(K) * [1280,960,1].';
corner3 = inv(K) * [0,960,1].';

projectedcameracenter  = reverseTvec * [cameracenter; 1];
projectedpixel0        = reverseTvec * [cameraspace0; 1];
projectedpixel1        = reverseTvec * [cameraspace1; 1];
projectedpixel2        = reverseTvec * [cameraspace2; 1];
projectedpixel3        = reverseTvec * [cameraspace3; 1];
projectedpixel4        = reverseTvec * [cameraspace4; 1];
projectedpixel5        = reverseTvec * [cameraspace5; 1];
projectedpixel6        = reverseTvec * [cameraspace6; 1];
projectedpixel7        = reverseTvec * [cameraspace7; 1];
projectedpixel8        = reverseTvec * [cameraspace8; 1];
projectedpixel9        = reverseTvec * [cameraspace9; 1];

projectedcorner0 = reverseTvec * [corner0; 1];
projectedcorner1 = reverseTvec * [corner1; 1];
projectedcorner2 = reverseTvec * [corner2; 1];
projectedcorner3 = reverseTvec * [corner3; 1];

rays0 = [projectedcameracenter(1:3,:).' ; projectedpixel0(1:3,:).'];
rays1 = [projectedcameracenter(1:3,:).' ; projectedpixel1(1:3,:).'];
rays2 = [projectedcameracenter(1:3,:).' ; projectedpixel2(1:3,:).'];
rays3 = [projectedcameracenter(1:3,:).' ; projectedpixel3(1:3,:).'];
rays4 = [projectedcameracenter(1:3,:).' ; projectedpixel4(1:3,:).'];
rays5 = [projectedcameracenter(1:3,:).' ; projectedpixel5(1:3,:).'];
rays6 = [projectedcameracenter(1:3,:).' ; projectedpixel6(1:3,:).'];
rays7 = [projectedcameracenter(1:3,:).' ; projectedpixel7(1:3,:).'];
rays8 = [projectedcameracenter(1:3,:).' ; projectedpixel8(1:3,:).'];
rays9 = [projectedcameracenter(1:3,:).' ; projectedpixel9(1:3,:).'];

plot3 (rays0(:,1), rays0(:,2), rays0(:,3)); hold on;
text  (projectedpixel0(1), projectedpixel0(2), projectedpixel0(3), 'Pt-1');
plot3 (rays1(:,1), rays1(:,2), rays1(:,3)); hold on;
text  (projectedpixel1(1), projectedpixel1(2), projectedpixel1(3), 'Pt-2'); 
plot3 (rays2(:,1), rays2(:,2), rays2(:,3)); hold on;
text  (projectedpixel2(1), projectedpixel2(2), projectedpixel2(3), 'Pt-3');
plot3 (rays3(:,1), rays3(:,2), rays3(:,3)); hold on;
text  (projectedpixel3(1), projectedpixel3(2), projectedpixel3(3), 'Pt-4');
plot3 (rays4(:,1), rays4(:,2), rays4(:,3)); hold on;
text  (projectedpixel4(1), projectedpixel4(2), projectedpixel4(3), 'Pt-5');
plot3 (rays5(:,1), rays5(:,2), rays5(:,3)); hold on;
text  (projectedpixel5(1), projectedpixel5(2), projectedpixel5(3), 'Pt-6');
plot3 (rays6(:,1), rays6(:,2), rays6(:,3)); hold on;
text  (projectedpixel6(1), projectedpixel6(2), projectedpixel6(3), 'Pt-7');
plot3 (rays7(:,1), rays7(:,2), rays7(:,3)); hold on;
text  (projectedpixel7(1), projectedpixel7(2), projectedpixel7(3), 'Pt-8');
plot3 (rays8(:,1), rays8(:,2), rays8(:,3)); hold on;
text  (projectedpixel8(1), projectedpixel8(2), projectedpixel8(3), 'Pt-9');
plot3 (rays9(:,1), rays9(:,2), rays9(:,3)); hold on;
text  (projectedpixel9(1), projectedpixel9(2), projectedpixel9(3), 'Pt-10');
text  (projectedcameracenter(1), projectedcameracenter(2), projectedcameracenter(3), 'Camera Center'); 

cornerpoints = [projectedcorner0 projectedcorner1 projectedcorner2 projectedcorner3];
x = cornerpoints(1, 1:4);
y = cornerpoints(2, 1:4);
z = cornerpoints(3, 1:4);
fill3 (x,y,z,'g');
labels = {'0,0','1280,960','0,960','1280,0'}; 
text (x,y,z, labels);

grid on;   
alpha(0.3);
