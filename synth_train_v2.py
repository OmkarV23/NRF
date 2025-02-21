import json
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import InstantNGPFieldRepresentation, forward_model
from torch.utils.tensorboard import SummaryWriter
import debugpy

debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()


# Initialize TensorBoard writer
writer = SummaryWriter()

f_operations = np.linspace(5.7e9, 8.2e9, 11)  # Operating frequencies
c = 3e8 

x_range = np.linspace(-99e-3, 99e-3, 31)
y_range = np.linspace(-99e-3, 99e-3, 41)
z_range = np.linspace(90e-3, 200e-3, 12)
X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")

x_pos = np.linspace(-99e-3, 99e-3, 12)
y_pos = np.linspace(-99e-3, 99e-3, 12)
transmitter = np.ones((len(x_pos) * len(y_pos), 2))
receiver = np.ones((len(x_pos) * len(y_pos), 2))

row_index = 0
for ii in range(len(x_pos)):
    for jj in range(len(y_pos)):
        transmitter[row_index, 0] = -0.06 - x_pos[ii]
        # transmitter[row_index, 0] = x_pos[ii]
        transmitter[row_index, 1] = y_pos[jj]
        receiver[row_index, 0] = 0.06 - x_pos[ii]
        # receiver[row_index, 0] = x_pos[ii]
        receiver[row_index, 1] = y_pos[jj]
        row_index += 1

z_range = np.linspace(90e-3, 200e-3, 12)
lz = len(z_range)
y_range = np.linspace(-99e-3, 99e-3, 41)
ly = len(y_range)
x_range = np.linspace(-99e-3, 99e-3, 31)
lx = len(x_range)
X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_range, y_range, z_range, indexing='ij')

sigma_volume = np.zeros((lx, ly, lz))

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

row_index = 0
Etf = []
Erf = []
for ii in range(144):
    xt, yt, zt = transmitter[ii, 0], transmitter[ii, 1], 0
    xr, yr, zr = receiver[ii, 0], receiver[ii, 1], 0

    r_nt = np.sqrt((X_mesh - xt)**2 + (Y_mesh - yt)**2 + (Z_mesh - zt)**2)
    r_nr = np.sqrt((X_mesh - xr)**2 + (Y_mesh - yr)**2 + (Z_mesh - zr)**2)

    for f_operation in f_operations:
        transmitted_field_f = np.exp(-1j * 2 * np.pi * f_operation / c * r_nt) / r_nt
        reflected_field_f = np.exp(-1j * 2 * np.pi * f_operation / c * r_nr) / r_nr

        Etf.append(transmitted_field_f.ravel())
        Erf.append(reflected_field_f.ravel())

H = np.array(Etf) * np.array(Erf)

g_gt = H @ sigma
magnitudes = np.abs(g_gt)
g_gt_normalized = g_gt / magnitudes.max() 

# between 0 to 1 

coords = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
coords_tensor = torch.tensor(coords, dtype=torch.float32).cuda()
H_tensor = torch.tensor(H, dtype=torch.complex64).cuda()
g_tensor = torch.tensor(g_gt_normalized, dtype=torch.complex64).cuda()
# g_tensor = torch.tensor(g_gt, dtype=torch.complex64).cuda()

# with open('network_config.json', 'r') as f:
#     config = json.load(f)

# model = InstantNGPFieldRepresentation(config).cuda()
model = InstantNGPFieldRepresentation().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=100,
#     T_mult=1,
#     eta_min=1e-4,
# )



loss_function = torch.nn.MSELoss()

#l1 loss
# loss_function = torch.nn.L1Loss()

# Training loop
num_epochs = 30000
progress_bar = tqdm(range(num_epochs), desc="Training Progress")

for epoch in progress_bar:
    optimizer.zero_grad()

    # Forward pass
    predictions = model(coords_tensor).squeeze(-1)
    g_pred = forward_model(predictions, H_tensor)

    # g_pred = g_pred * 2000 ########## scaling value

    # g_pred will be unbounded

    g_pred = g_pred[:g_tensor.shape[0]]

    # Compute loss
    real_loss = loss_function(g_pred.real, g_tensor.real)
    imag_loss = loss_function(g_pred.imag, g_tensor.imag)
    loss = real_loss + imag_loss #+ torch.norm(predictions, p=1) / predictions.shape[0]

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # if epoch % 100 == 0:
    #     scheduler.base_lrs[0] = scheduler.base_lrs[0] * 0.5

    # scheduler.step(epoch)
    # current_lr = optimizer.param_groups[0]['lr']


    # Log losses to TensorBoard
    writer.add_scalar('Real Loss', real_loss.item(), epoch)
    writer.add_scalar('Imaginary Loss', imag_loss.item(), epoch)
    writer.add_scalar('Total Loss', loss.item(), epoch)
    # writer.add_scalar('Learning Rate', current_lr, epoch)

    # Update progress bar
    progress_bar.set_postfix({'Real Loss': real_loss.item(), 'Imaginary Loss': imag_loss.item(), 'Total Loss': loss.item()})

    # Save checkpoints every 500 epochs
    if (epoch + 1) % 100 == 0:
        checkpoint_path = f'synthetic_data_bistatic/model_2_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch+1}")

# Save final model
torch.save(model.state_dict(), 'synthetic_data_bistatic/model_2_final.pth')
print("Final model saved.")

writer.close()

