function ply_to_pcd()
% This function converts the ply file into pcd format which is employed in
% PCL library

[filename, pathname] = uigetfile({'*.ply;', 'Choose the camera position from bundler'});

plyfile = fopen([pathname filename], 'r');
pcdfile = fopen([pathname filename(1:end-4) '.pcd'], 'wt');

% ingore the ply header
plytline = fgetl(plyfile);

% Write the pcd header
fprintf(pcdfile, '%s\n', '# .PCD v.7 - Point Cloud Data file format');

% ingore the ply format
plytline = fgetl(plyfile);

% Write the pcd version
fprintf(pcdfile, '%s\n', 'VERSION .7');

% ingore the ply comment
plytline = fgetl(plyfile);

% grab the number of vertices in ply
% plytline = fgetl(plyfile);
NumVertices = sscanf(plytline, ['element vertex' '%d']);
plytline = fgetl(plyfile);

% analysis information constitution in ply file
% throw away x, y, z properties in header
plytline = fgetl(plyfile);
plytline = fgetl(plyfile);
plytline = fgetl(plyfile);

plytline = fgetl(plyfile);
disp('===========================');
disp('                           ');
if ~isempty(strfind(plytline, 'red'))
    tic
    fprintf('PLY file has color information for vertex!\n');
    % throw away color description in header
    for k = 1:6
        plytline = fgetl(plyfile);
    end
    % write the header in pcd file
    fprintf(pcdfile, '%s\n', 'FIELDS x y z rgb');
    fprintf(pcdfile, '%s\n', 'SIZE 4 4 4 4');
    fprintf(pcdfile, '%s\n', 'TYPE F F F U');
    fprintf(pcdfile, '%s\n', 'COUNT 1 1 1 1');
    fprintf(pcdfile, 'WIDTH %d\n', NumVertices);
    fprintf(pcdfile, 'HEIGHT %d\n', 1);
    fprintf(pcdfile, '%s\n', 'VIEWPOINT 0 0 0 1 0 0 0');
    fprintf(pcdfile, 'POINTS %d\n', NumVertices);
    fprintf(pcdfile, '%s\n', 'DATA ascii');
    for iter = 1:NumVertices
        plytline = fgetl(plyfile);
        data = sscanf(plytline, '%f');
        % encode the rgb through bitshifting
        red = dec2hex(data(4));
        if data(4) < 16
            red = ['0' red];
        end
        green = dec2hex(data(5));
        if data(5) < 16
            green = ['0' green];
        end
        blue = dec2hex(data(6));
        if data(6) < 16
            blue = ['0' blue];
        end
%        alpha = dec2hex(data(7));
        rgb = hex2dec([red green blue]);
        % write the x, y, z and rgb into pcd file
        fprintf(pcdfile,'%5.8f %5.8f %5.8f %u\n',data(1), data(2), data(3), rgb);
    end
    fprintf('Format conversion complete!\n');
    toc
else
    tic
    fprintf('PLY file has no color information for vertex!\n');
    % throw away color description in header
    for k = 1:2
        plytline = fgetl(plyfile);
    end
    % write the header in pcd file
    fprintf(pcdfile, '%s\n', 'FIELDS x y z');
    fprintf(pcdfile, '%s\n', 'SIZE 4 4 4');
    fprintf(pcdfile, '%s\n', 'TYPE F F F');
    fprintf(pcdfile, '%s\n', 'COUNT 1 1 1');
    fprintf(pcdfile, 'WIDTH %d\n', NumVertices);
    fprintf(pcdfile, 'HEIGHT %d\n', 1);
    fprintf(pcdfile, '%s\n', 'VIEWPOINT 0 0 0 1 0 0 0');
    fprintf(pcdfile, 'POINTS %d\n', NumVertices);
    fprintf(pcdfile, '%s\n', 'DATA ascii');
    for iter = 1:NumVertices
        plytline = fgetl(plyfile);
        data = sscanf(plytline, '%f');
        % write the x, y, z into pcd file
        fprintf(pcdfile,'%5.8f %5.8f %5.8f\n',data(1), data(2), data(3));
    end
    fprintf('Format conversion complete!\n');
    toc
end

fclose(plyfile);
fclose(pcdfile);

    
        