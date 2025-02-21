clc; clear; close all;

% Define constants
f_operations = linspace(5.7e9, 8.2e9, 11);
c = 3e8; 
lambda = c ./ f_operations;
k = 2 * pi ./ lambda; 

% Define transmitter and receiver positions
x_pos = linspace(-99e-3, 99e-3, 12);
y_pos = linspace(-99e-3, 99e-3, 12);
transmitter = ones(length(x_pos) * length(y_pos), 2);
receiver = ones(length(x_pos) * length(y_pos), 2);
row_index = 1;

for ii = 1:length(x_pos)
    for jj = 1:length(y_pos)
        transmitter(row_index, 1) = -0.06 - x_pos(ii);
        transmitter(row_index, 2) = y_pos(jj);
        receiver(row_index, 1) = 0.06 - x_pos(ii);
        receiver(row_index, 2) = y_pos(jj);
        row_index = row_index + 1;
    end
end

% Define spatial grid for reconstruction
z_range = linspace(90e-3, 200e-3, 12); lz = length(z_range);
y_range = linspace(-99e-3, 99e-3, 41); ly = length(y_range);
x_range = linspace(-99e-3, 99e-3, 31); lx = length(x_range);
[Y_mesh, X_mesh, Z_mesh] = meshgrid(y_range, x_range, z_range);

% Define two rectangles in the 3D volume
sigma_volume = zeros(lx, ly, lz); % Initialize a 3D scatterer volume

% First rectangle bounds
rect1_x = [-30e-3, -10e-3]; % X-bounds
rect1_y = [-40e-3, 0e-3];   % Y-bounds
rect1_z = [110e-3, 140e-3]; % Z-bounds

% Second rectangle bounds
rect2_x = [10e-3, 30e-3];   % X-bounds
rect2_y = [10e-3, 50e-3];   % Y-bounds
rect2_z = [150e-3, 180e-3]; % Z-bounds

% Add the first rectangle
sigma_volume( ...
    (x_range >= rect1_x(1) & x_range <= rect1_x(2)), ...
    (y_range >= rect1_y(1) & y_range <= rect1_y(2)), ...
    (z_range >= rect1_z(1) & z_range <= rect1_z(2))) = 1;

% Add the second rectangle
sigma_volume( ...
    (x_range >= rect2_x(1) & x_range <= rect2_x(2)), ...
    (y_range >= rect2_y(1) & y_range <= rect2_y(2)), ...
    (z_range >= rect2_z(1) & z_range <= rect2_z(2))) = 1;

% Flatten the 3D scatterer volume into the scatterer vector sigma
sigma = sigma_volume(:);

% Forward simulation
row_index = 1;
for ii = 1:144
    xt = transmitter(ii, 1); yt = transmitter(ii, 2); zt = 0;
    xr = receiver(ii, 1); yr = receiver(ii, 2); zr = 0;

    r_nt = sqrt((X_mesh - xt).^2 + (Y_mesh - yt).^2 + (Z_mesh - zt).^2);
    r_nr = sqrt((X_mesh - xr).^2 + (Y_mesh - yr).^2 + (Z_mesh - zr).^2);

    for f_index = 1:length(f_operations)
        f_operation = f_operations(f_index);
        transmitted_field_f = exp(-1j * 2 * pi * f_operation / c * r_nt) ./ r_nt;
        reflected_field_f = exp(-1j * 2 * pi * f_operation / c * r_nr) ./ r_nr;

        Etf(row_index, :) = reshape(transmitted_field_f, 1, []);
        Erf(row_index, :) = reshape(reflected_field_f, 1, []);
        row_index = row_index + 1;
    end
end
H_forward = Etf .* Erf;

% Simulated observation
g = H_forward * sigma;

% Backward simulation
row_index = 1;
for ii = 1:144
    xt = transmitter(ii, 1); yt = transmitter(ii, 2); zt = 0;
    xr = receiver(ii, 1); yr = receiver(ii, 2); zr = 0;

    r_nt = sqrt((X_mesh - xt).^2 + (Y_mesh - yt).^2 + (Z_mesh - zt).^2);
    r_nr = sqrt((X_mesh - xr).^2 + (Y_mesh - yr).^2 + (Z_mesh - zr).^2);

    for f_index = 1:length(f_operations)
        f_operation = f_operations(f_index);
        transmitted_field_b = exp(-1j * 2 * pi * f_operation / c * r_nt) ./ r_nt;
        reflected_field_b = exp(-1j * 2 * pi * f_operation / c * r_nr) ./ r_nr;

        Etb(row_index, :) = reshape(transmitted_field_b, 1, []);
        Erb(row_index, :) = reshape(reflected_field_b, 1, []);
        row_index = row_index + 1;
    end
end
H_Backward = Etb .* Erb;

% Reconstruction using matched filter
sigma_estimated_mf = H_Backward' * g;
sigma_reshaped_mf = reshape(sigma_estimated_mf, [lx, ly, lz]);

% Visualization: 2D slice
z_slice = 7; % Slice of interest
figure;
imagesc(x_range * 100, y_range * 100, abs(sigma_reshaped_mf(:, :, z_slice))');
colorbar;
xlabel('x (cm)');
ylabel('y (cm)');
axis xy;
title(['Reconstructed Two Rectangles (Matched Filter) at ',num2str(z_range(z_slice)*10^3),' mm']);

% % Visualization: 3D plot
% figure;
% isosurface(X_mesh, Y_mesh, Z_mesh, abs(sigma_reshaped_mf), max(abs(sigma_reshaped_mf(:))) * 0.5);
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% title('3D Reconstruction of Two Rectangles');
% grid on;
% axis equal;
% colormap(jet);
% colorbar;

%%
% Extract the indices of non-zero elements in sigma_volume
[x_idx, y_idx, z_idx] = ind2sub(size(sigma_volume), find(sigma_volume));

% Map indices to coordinates
x_coords = x_range(x_idx); % X-coordinates of scatterers
y_coords = y_range(y_idx); % Y-coordinates of scatterers
z_coords = z_range(z_idx); % Z-coordinates of scatterers

% Create the 3D scatter plot
figure;
scatter3(x_coords * 1000, y_coords * 1000, z_coords * 1000, 50, 'filled'); % Scale to mm
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
xlim([min(x_range) max(x_range)] * 1000); % Set X axis to the full volume
ylim([min(y_range) max(y_range)] * 1000); % Set Y axis to the full volume
zlim([min(z_range) max(z_range)] * 1000); % Set Z axis to the full volume
title('3D Scatter Plot of Scatterer Volume Over Full Grid');
grid on;
axis equal;

