import torch
import tinycudann as tcnn
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from process_data import process_radar_data
import pyvista as pv
from model import InstantNGPFieldRepresentation, forward_model

model = InstantNGPFieldRepresentation().cuda()
checkpoint = torch.load('/home/omkar/Desktop/All_Desktop_files/Projetcs/Neural_Radar_fields/synthetic_data_bistatic/model_2_epoch_20000.pth')
model.load_state_dict(checkpoint)
model.eval()


def plot_3D_points(X, Y, Z, f_est_3D, threshold_factor=0.01, cmap='coolwarm'):
    """Scatter plot of 3D points based on intensity threshold"""
    
    f_est_cube = np.abs(f_est_3D) ** 2  # Compute squared magnitude
    isovalue = f_est_cube.max() * threshold_factor  # Define threshold level

    # Get indices where the intensity is above the threshold
    mask = f_est_cube >= isovalue

    # Extract corresponding coordinates
    X_pts, Y_pts, Z_pts, F_vals = X[mask], Y[mask], Z[mask], f_est_cube[mask]

    # Plot using scatter3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    scatter = ax.scatter(X_pts, Y_pts, Z_pts, c=F_vals, cmap=cmap, s=10)
    
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Intensity")

    # Set labels and title
    ax.set_xlabel("X-axis (m)")
    ax.set_ylabel("Y-axis (m)")
    ax.set_zlabel("Z-axis (m)")
    ax.set_title("3D Point Cloud Visualization")

    plt.show()

x_range = np.linspace(-99e-3, 99e-3, 31)
y_range = np.linspace(-99e-3, 99e-3, 41)
z_range = np.linspace(90e-3, 200e-3, 12)
X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")

sigma_volume = np.zeros((len(x_range), len(y_range), len(z_range)))

rect1_x = [-30e-3, -10e-3]
rect1_y = [-40e-3, 0e-3]
rect1_z = [110e-3, 140e-3]

rect2_x = [10e-3, 30e-3]
rect2_y = [10e-3, 50e-3]
rect2_z = [150e-3, 180e-3]


x_indices = np.where((x_range >= rect1_x[0]) & (x_range <= rect1_x[1]))[0]
y_indices = np.where((y_range >= rect1_y[0]) & (y_range <= rect1_y[1]))[0]
z_indices = np.where((z_range >= rect1_z[0]) & (z_range <= rect1_z[1]))[0]

sigma_volume[np.ix_(x_indices, y_indices, z_indices)] = 1


x_indices = np.where((x_range >= rect2_x[0]) & (x_range <= rect2_x[1]))[0]
y_indices = np.where((y_range >= rect2_y[0]) & (y_range <= rect2_y[1]))[0]
z_indices = np.where((z_range >= rect2_z[0]) & (z_range <= rect2_z[1]))[0]
sigma_volume[np.ix_(x_indices, y_indices, z_indices)] = 1


sigma = sigma_volume.ravel()


coords = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
coords_tensor = torch.tensor(coords, dtype=torch.float32).cuda()


# x_pos = np.linspace(-99e-3, 99e-3, 12)
# y_pos = np.linspace(-99e-3, 99e-3, 12)
# transmitter = np.ones((len(x_pos) * len(y_pos), 2))
# receiver = np.ones((len(x_pos) * len(y_pos), 2))


# ### seting up tx and rx

# row_index = 0
# for ii in range(len(x_pos)):
#     for jj in range(len(y_pos)):
#         transmitter[row_index, 0] = -0.06 - x_pos[ii]
#         # transmitter[row_index, 0] = x_pos[ii]
#         transmitter[row_index, 1] = y_pos[jj]
#         receiver[row_index, 0] = 0.06 - x_pos[ii]
#         # receiver[row_index, 0] = x_pos[ii]
#         receiver[row_index, 1] = y_pos[jj]
#         row_index += 1

# f_operations = np.linspace(5.7e9, 8.2e9, 11)  # Operating frequencies
# c = 3e8 

# # calculating H matrix


# row_index = 0
# Etf = []
# Erf = []
# for ii in range(144):
#     xt, yt, zt = transmitter[ii, 0], transmitter[ii, 1], 0
#     xr, yr, zr = receiver[ii, 0], receiver[ii, 1], 0

#     r_nt = np.sqrt((X - xt)**2 + (Y - yt)**2 + (Z - zt)**2)
#     r_nr = np.sqrt((X - xr)**2 + (Y - yr)**2 + (Z - zr)**2)

#     for f_operation in f_operations:
#         transmitted_field_f = np.exp(-1j * 2 * np.pi * f_operation / c * r_nt) / r_nt
#         reflected_field_f = np.exp(-1j * 2 * np.pi * f_operation / c * r_nr) / r_nr

#         Etf.append(transmitted_field_f.ravel())
#         Erf.append(reflected_field_f.ravel())

# H = np.array(Etf) * np.array(Erf)


# g_gt = H @ sigma_volume.ravel()
# sigma_match_f = np.linalg.pinv(H.conj()) @ g_gt

# Forward pass through the model
f_est = model(coords_tensor)
f_est = f_est.cpu().detach().numpy()
f_est = f_est.reshape(len(x_range), len(y_range), len(z_range))




plot_3D_points(X, Y, Z, f_est, threshold_factor=0.01, cmap='coolwarm')
plot_3D_points(X, Y, Z, sigma_volume, threshold_factor=0.01, cmap='viridis')
# plot_3D_points(X, Y, Z, np.abs(sigma_match_f), threshold_factor=0.01, cmap='viridis')
