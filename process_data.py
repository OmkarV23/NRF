import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio

def process_radar_data(copper_tape_file, probe_response_file):
    mat_contents = sio.loadmat(copper_tape_file)
    data = mat_contents['data']
    
    Z = np.linspace(90e-3, 200e-3, 12)
    Y = np.linspace(-99e-3, 99e-3, 41)
    X = np.linspace(-99e-3, 99e-3, 31)
    
    Y_mesh, X_mesh, Z_mesh = np.meshgrid(Y, X, Z)
    
    X_meshp = X_mesh.T.reshape(-1)
    Y_meshp = Y_mesh.T.reshape(-1)
    Z_meshp = Z_mesh.T.reshape(-1)
    
    z_TX = 0
    z_RX = 0
    
    c = 2.998e8
    num_freq = 11
    freq_space = 100 / (num_freq - 1)
    f = np.linspace(5.7e9, 8.2e9, num_freq)
    lambda_ = c / f
    k = 2 * np.pi / lambda_
    
    x_scan = data['X'][0][0].T.reshape(-1)
    y_scan = data['Y'][0][0].T.reshape(-1)
    
    H = np.zeros((num_freq * len(x_scan), X_meshp.size), dtype=complex)
    
    print("Calculating sensing matrix...")
    for ii in range(len(x_scan)):
        if ii % 10 == 0:
            print(f"Processing scan position {ii+1}/{len(x_scan)}")
        for jj in range(num_freq):
            x_TX = -0.06 - x_scan[ii] * 1e-3
            y_TX = -y_scan[ii] * 1e-3
            x_RX = 0.06 - x_scan[ii] * 1e-3
            y_RX = -y_scan[ii] * 1e-3
            
            r_TX2ROI = np.sqrt((x_TX - X_meshp)**2 + (y_TX - Y_meshp)**2 + (Z_meshp - z_TX)**2)
            r_RX2ROI = np.sqrt((x_RX - X_meshp)**2 + (y_RX - Y_meshp)**2 + (Z_meshp - z_RX)**2)
            
            H[ii * num_freq + jj, :] = np.exp(-1j * k[jj] * r_TX2ROI) * np.exp(-1j * k[jj] * r_RX2ROI)
    
    plt.figure()
    plt.imshow(np.angle(H), aspect='auto')
    plt.colorbar()
    plt.title('Angle of H')
    plt.xlabel('Data Points')
    plt.ylabel('Measurement Index')
    plt.show()

    probe_data = sio.loadmat(probe_response_file)
    ProbeResponse2 = probe_data['ProbeResponse2']
    ProbeResponse2 = ProbeResponse2[0:401:4]
    measurements = data['measurements'][0][0]

    freq_indices = np.arange(0, 101, int(freq_space)).astype(int)
    g_gt = np.zeros(num_freq * 12 * 12, dtype=complex)
    
    print("Processing measurements...")
    for ii in range(12):
        for jj in range(12):
            idx_start = (ii * num_freq * 12) + (jj * num_freq)
            idx_end = idx_start + num_freq
            measurement = measurements[jj, ii, freq_indices]
            probe_response = np.exp(1j * 2 * np.angle(ProbeResponse2[freq_indices])).reshape(-1)
            g_gt[idx_start:idx_end] = measurement / probe_response
        
    return X, Y, Z, H, g_gt