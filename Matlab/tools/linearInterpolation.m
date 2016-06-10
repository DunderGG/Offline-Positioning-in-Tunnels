%% linear interpolation plotting

A = [0,0];
B = [2,2];

vec_AB = B-A;

line = [A; B];

r = 1;
d = 0.5;

L = 2 * sqrt(r^2 - d^2);
L_half = L/2;

B_a = B + L / sqrt(vec_AB(1)^2 + vec_AB(2)^2) * vec_AB;

figure;
plot (line(:,1), line(:,2)); hold on;
plot (B_a(1), B_a(2), 'x'); hold on;
plot (line(:,1), line(:,2), 'x'); hold on;

% circle

th = 0:pi/50:2*pi;
xunit = r * cos(th) + B(1);
yunit = r * sin(th) + B(2);

xunit2 = r * cos(th) + B_a(1);
yunit2 = r * sin(th) + B_a(2);

plot(xunit, yunit); hold on;
plot(xunit2, yunit2);
