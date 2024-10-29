clc  
clear     

load('./CopperTape_15cm.mat')

% Define grid points for the 3D region of interest (ROI) in space
Z = linspace(90e-3,200e-3,12);     % Z axis (depth) from 90 mm to 200 mm (12 points)
Y = linspace(-99e-3,99e-3,41);     % Y axis from -99 mm to 99 mm (41 points)
X = linspace(-99e-3,99e-3,31);     % X axis from -99 mm to 99 mm (31 points)

% Create 3D meshgrid for the X, Y, and Z coordinates
[Y_mesh, X_mesh, Z_mesh] = meshgrid(Y, X, Z);

% Reshape the 3D mesh grid to 1D arrays
X_meshp = reshape(X_mesh,[1, numel(Z)*numel(Y)*numel(X)]);
Y_meshp = reshape(Y_mesh,[1, numel(Z)*numel(Y)*numel(X)]);
Z_meshp = reshape(Z_mesh,[1, numel(Z)*numel(Y)*numel(X)]);

% Set the z-coordinates for the transmitter (TX) and receiver (RX) to zero
z_TX = 0;
z_RX = 0;

% Define constants and parameters for frequency and wave calculations
c = 2.998e8;                       
num_freq = 11;                     % Number of frequency points
freq_space = 100/(num_freq-1);      % Frequency spacing
f = linspace(5.7e9,8.2e9,num_freq); 
lambda = c ./ f;                    % Calculate wavelength for each frequency
k = 2*pi ./ lambda;                 % Calculate wavenumber for each frequency

% Reshape the scanning positions (X and Y) from the measurement data
x_scan = reshape(data.X,[144,1]);
y_scan = reshape(data.Y,[144,1]);

%% Loop through each scanning position and frequency to calculate the sensing matrix
for ii = 1:numel(x_scan)
    for jj = 1:num_freq

        % Calculate the TX and RX positions for each scan point
        x_TX = -0.06 - x_scan(ii)*10^-3;  
        y_TX = -y_scan(ii)*10^-3;        
        x_RX = 0.06 - x_scan(ii)*10^-3;  
        y_RX = -y_scan(ii)*10^-3;        

        % Calculate the distances from TX and RX to each point in the ROI
        r_TX2ROI = sqrt((x_TX - X_meshp).^2 + (y_TX - Y_meshp).^2 + (Z_meshp - z_TX).^2);
        r_RX2ROI = sqrt((x_RX - X_meshp).^2 + (y_RX - Y_meshp).^2 + (Z_meshp - z_RX).^2);

        % Compute the sensing matrix for the current scan and frequency
        H(((ii-1)*num_freq + jj), :) = exp(-1i .* k(jj) .* r_TX2ROI) .* exp(-1i .* k(jj) .* r_RX2ROI);
    end
end

imagesc((angle(H)))  
colorbar;      

%% Load and process the probe response data
load 'ProbeResponse2.mat'         
ProbeResponse2 = ProbeResponse2(1:4:401); 

% Loop through each measurement point and frequency to process the measured data
for ii = 1:12
    for jj = 1:12
        % Extract the measurements and normalize by the probe response
        g(((ii-1)*num_freq*12+(jj-1)*num_freq+1) : ((ii-1)*num_freq*12+(jj*num_freq))) = squeeze(data.measurements(jj, ii, 1:freq_space:101)) ./ (exp(1j*2*angle(ProbeResponse2(1:freq_space:101))));
    end
end

% Matched filter
f_est = -H' * transpose(g);

% Reshape the estimated field into a 3D array
f_est_reshape = reshape(f_est, [numel(X), numel(Y), numel(Z)]);

figure
imagesc(X*1000, Y*1000, abs(transpose(f_est_reshape(:, :, 8))).^2) 
camroll(180) 
cb = colorbar; 
title('15 cm Copper Tape')  
set(gca, "FontSize", 14)
xlabel('X (mm)') 
ylabel('Y (mm)')  
ylabel(cb, 'f_{estimate}^2', 'Rotation', 270)  
ax = gca;
ax.XAxisLocation = 'top';   
ax.YAxisLocation = 'right';
ax.XDir = 'reverse';     
ax.YDir = 'reverse';     

%% 3D plot
% Assuming f_est_reshape is your 3D field estimate with shape (31, 41, 12)
f_est_cube = abs(f_est_reshape).^2;  % Square of the magnitude of the field estimate

% Directly use X_mesh, Y_mesh, Z_mesh since their sizes already match f_est_cube
% X_grid = X_mesh * 1000;  % Convert to mm
% Y_grid = Y_mesh * 1000;  % Convert to mm
% Z_grid = Z_mesh * 1000;  % Convert to mm
[X_grid, Y_grid, Z_grid] = meshgrid(X * 1000, Y * 1000, Z * 1000);

% Define an isovalue for the isosurface (a threshold for the 3D rendering)
isovalue = max(f_est_cube(:)) * 0.5;  % Adjust threshold as needed

% Plot the isosurface
figure;
p = patch(isosurface(X_grid, Y_grid, Z_grid, f_est_cube, isovalue));
isonormals(X_grid, Y_grid, Z_grid, f_est_cube, p);
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');

% Add additional isocaps for a better sense of depth
patch(isocaps(X_grid, Y_grid, Z_grid, f_est_cube, isovalue), ...
    'FaceColor', 'interp', 'EdgeColor', 'none');

% Set view and labels
view(3);
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
axis tight;
camlight;
lighting gouraud;
title('3D Isosurface - 15 cm Copper Tape');
colormap('hot');
colorbar;
