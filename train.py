import json
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from process_data import process_radar_data
from model import InstantNGPFieldRepresentation, forward_model
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

X, Y, Z, H, g_gt = process_radar_data('CopperTape_15cm.mat', 'ProbeResponse2.mat')
coords = np.stack(np.meshgrid(X, Y, Z), -1).reshape(-1, 3)

coords_tensor = torch.tensor(coords, dtype=torch.float32).cuda()
H = torch.tensor(H, dtype=torch.complex64).cuda()
g_gt = torch.tensor(g_gt, dtype=torch.complex64).cuda()

with open('network_config.json', 'r') as f:
    config = json.load(f)

model = InstantNGPFieldRepresentation(config).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

num_epochs = 5000
progress_bar = tqdm(range(num_epochs), desc="Training Progress")

for epoch in progress_bar:
    optimizer.zero_grad()
    
    predictions, normals_ = model(coords_tensor)
    g_pred = forward_model(predictions, H)

    real_loss = loss_function(g_pred.real, g_gt.real)
    imag_loss = loss_function(g_pred.imag, g_gt.imag)
    loss = real_loss + imag_loss
    
    loss.backward()
    optimizer.step()

    writer.add_scalar('Real Loss', real_loss.item(), epoch)
    writer.add_scalar('Imaginary Loss', imag_loss.item(), epoch)
    writer.add_scalar('Total Loss', loss.item(), epoch)

    progress_bar.set_postfix({'Real Loss': real_loss.item(), 'Imaginary Loss': imag_loss.item(), 'Total Loss': loss.item()})
    
    if (epoch + 1) % 500 == 0:
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch+1}")

torch.save(model.state_dict(), 'checkpoints/model_final.pth')
print("Final model saved.")

writer.close()

# # Plot the angle of the predicted field
# plt.figure()
# plt.imshow(np.angle(predictions.detach().cpu().numpy()).reshape(31, 41, 12)[:, :, 0], aspect='auto')
# plt.colorbar()
# plt.title('Angle of Predicted Field')
# plt.xlabel('Y')
# plt.ylabel('X')
# plt.show()

    
