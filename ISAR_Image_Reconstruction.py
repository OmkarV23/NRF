import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio
# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()

def process_radar_data(copper_tape_file, probe_response_file):
    # Load the .mat files
    mat_contents = sio.loadmat(copper_tape_file)
    data = mat_contents['data']
    
    # Define grid points for the 3D region of interest (ROI) in space
    Z = np.linspace(90e-3, 200e-3, 12)    # Z axis (depth) from 90mm to 200mm
    Y = np.linspace(-99e-3, 99e-3, 41)    # Y axis from -99mm to 99mm
    X = np.linspace(-99e-3, 99e-3, 31)    # X axis from -99mm to 99mm
    
    # Create 3D meshgrid for the X, Y, and Z coordinates
    Y_mesh, X_mesh, Z_mesh = np.meshgrid(Y, X, Z)
    
    # Reshape the 3D mesh grid to 1D arrays
    X_meshp = X_mesh.T.reshape(-1)
    Y_meshp = Y_mesh.T.reshape(-1)
    Z_meshp = Z_mesh.T.reshape(-1)
    
    # Set the z-coordinates for the transmitter (TX) and receiver (RX)
    z_TX = 0
    z_RX = 0
    
    # Define constants and parameters - using 11 frequency points
    c = 2.998e8                        # Speed of light
    num_freq = 11
    freq_space = 100 / (num_freq - 1)
    f = np.linspace(5.7e9, 8.2e9, num_freq)  # Frequency range
    lambda_ = c / f                    # Wavelengths
    k = 2 * np.pi / lambda_            # Wave numbers
    
    # Extract and reshape scanning positions
    x_scan = data['X'][0][0].T.reshape(-1)
    y_scan = data['Y'][0][0].T.reshape(-1)
    
    # Initialize sensing matrix H
    H = np.zeros((num_freq * len(x_scan), X_meshp.size), dtype=complex)
    
    print("Calculating sensing matrix...")
    # Calculate sensing matrix H
    for ii in range(len(x_scan)):
        if ii % 10 == 0:  # Progress indicator
            print(f"Processing scan position {ii+1}/{len(x_scan)}")
        for jj in range(num_freq):
            # Calculate TX and RX positions
            x_TX = -0.06 - x_scan[ii] * 1e-3
            y_TX = -y_scan[ii] * 1e-3
            x_RX = 0.06 - x_scan[ii] * 1e-3
            y_RX = -y_scan[ii] * 1e-3
            
            # Calculate distances
            r_TX2ROI = np.sqrt((x_TX - X_meshp)**2 + (y_TX - Y_meshp)**2 + (Z_meshp - z_TX)**2)
            r_RX2ROI = np.sqrt((x_RX - X_meshp)**2 + (y_RX - Y_meshp)**2 + (Z_meshp - z_RX)**2)
            
            # Compute sensing matrix elements
            H[ii * num_freq + jj, :] = np.exp(-1j * k[jj] * r_TX2ROI) * np.exp(-1j * k[jj] * r_RX2ROI)
    
    # Visualize the angle of H
    plt.figure()
    plt.imshow(np.angle(H), aspect='auto')
    plt.colorbar()
    plt.title('Angle of H')
    plt.xlabel('Data Points')
    plt.ylabel('Measurement Index')
    plt.show()

    # Load and process probe response
    probe_data = sio.loadmat(probe_response_file)
    ProbeResponse2 = probe_data['ProbeResponse2']
    ProbeResponse2 = ProbeResponse2[0:401:4]
    measurements = data['measurements'][0][0]

    freq_indices = np.arange(0, 101, int(freq_space)).astype(int)
    
    # Initialize measurement vector g
    g = np.zeros(num_freq * 12 * 12, dtype=complex)
    
    print("Processing measurements...")
    # Process measurements at specific frequency indices
    for ii in range(12):
        for jj in range(12):
            idx_start = (ii * num_freq * 12) + (jj * num_freq)
            idx_end = idx_start + num_freq

            # Extract the measurements at specific frequency indices
            measurement = measurements[jj, ii, freq_indices]
            probe_response = np.exp(1j * 2 * np.angle(ProbeResponse2[freq_indices])).reshape(-1)

            # Normalize measurements by the probe response
            g[idx_start:idx_end] = measurement / probe_response
    
    print("Applying matched filter...")
    # Apply matched filter
    f_est = -H.conj().T @ g
    
    # Reshape result to 3D
    f_est_3D = np.reshape(f_est, (len(X), len(Y), len(Z)), order='F')
    
    return X, Y, Z, f_est_3D

def plot_3D_results(X, Y, Z, f_est_3D, threshold_factor=0.5):
    """Plot 3D visualization of the results"""
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Compute the magnitude squared of the field estimate
    f_est_cube = np.abs(f_est_3D) ** 2

    # Define an isovalue for the isosurface (a threshold for the 3D rendering)
    isovalue = f_est_cube.max() * threshold_factor  # Adjust threshold as needed

    # Compute the isosurface using marching cubes
    spacing = (X[1] - X[0]) * 1000, (Y[1] - Y[0]) * 1000, (Z[1] - Z[0]) * 1000
    verts, faces, normals, values = measure.marching_cubes(f_est_cube, level=isovalue, spacing=spacing)

    # Plot the isosurface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor('red')
    ax.add_collection3d(mesh)

    # Set view and labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # Set limits
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    plt.title('3D Isosurface - 15 cm Copper Tape')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, Y, Z, f_est_3D = process_radar_data('CopperTape_15cm.mat', 'ProbeResponse2.mat')

    # Visualize the estimated field at a specific depth (e.g., Z index 7)
    plt.figure(figsize=(10, 8))

    # Corrected index and transpose
    img_data = np.abs(f_est_3D[:, :, 7].T) ** 2

    # Flip the data to match MATLAB's camroll(180)
    img_data_flipped = np.flipud(np.fliplr(img_data))

    # Adjust extent to reverse axes
    extent = [X[-1]*1000, X[0]*1000, Y[-1]*1000, Y[0]*1000]

    # Plot the image with origin set to 'lower'
    plt.imshow(img_data_flipped, extent=extent, origin='lower', cmap='hot', aspect='auto')

    # Move x-axis to the top
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel('X (mm)')

    # Move y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    plt.ylabel('Y (mm)')

    # Set title and colorbar
    plt.title('15 cm Copper Tape')
    plt.colorbar(label='$f_{estimate}^2$')
    plt.show()

    plot_3D_results(X, Y, Z, f_est_3D, threshold_factor=0.5)