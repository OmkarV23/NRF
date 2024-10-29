# InstantNGP Field Representation for Radar Data Processing

This repository implements an Instant Neural Graphics Primitives (InstantNGP) model using `tiny-cuda-nn` for predicting field estimates over a 3D grid. The model computes measurements using a forward model with complex values, designed for applications in radar data processing. The model includes a neural representation using multi-resolution hash encoding and computes field estimates based on a learned function.

## Features

- **Multi-resolution Hash Encoding**: The model utilizes efficient multi-level hash encoding for high-resolution spatial representations.
- **Complex-Valued Forward Model**: The model computes complex field estimates and includes a forward model to predict measurements.
- **Normal Vector Calculation**: The model calculates normal vectors at each point in the 3D grid, though these are not currently used in optimization.
- **Extensibility for Scattering and Transmission Models**:
  - **Lambertian Scattering Model**: The computed normals can be used in a Lambertian scattering model in future extensions.
  - **Transmission Probabilities**: Transmission probabilities based on predicted scattering values can also be computed but are not included in this version.

## Requirements

To run the code, you will need the following libraries:
- `torch` (PyTorch)
- `tinycudann` (Tiny CUDA Neural Networks)

Install these requirements with the following command:
```bash
pip install torch
pip install git+https://github.com/NVlabs/tiny-cuda-nn.git
```

## Code Overview

### Key Components

1. `InstantNGPFieldRepresentation`:
   - This is the main model, which includes multi-resolution hash encoding and an MLP with sine activation.
   - The model takes 3D coordinates as input and outputs complex-valued field estimates.

2. `forward_model`:
   - The forward model computes measurements based on the predicted field estimates by multiplying with the conjugate transpose of the sensing matrix H.

3. **Normal Calculation**:
   - Normal vectors are calculated as gradients of the model's output with respect to the input coordinates.
   - These normals are currently not used in optimization but can be incorporated into the forward model for a **Lambertian scattering model**.

4. **Potential Extensions**:
   - **Lambertian Scattering Model**: Normals could be used to model scattering based on Lambertian reflection principles.
   - **Transmission Probabilities**: Transmission probabilities could be computed based on predicted scattering values to simulate transmission effects.

### File Structure

- `model.py`: Defines the `InstantNGPFieldRepresentation` model.
- `process_data.py`: Contains data processing functions (not included here).
- `forward_model` function: Computes measurements from field estimates based on the sensing matrix H.

## Usage Example

```python
import torch
from model import InstantNGPFieldRepresentation, forward_model

# Initialize model with default configuration
model = InstantNGPFieldRepresentation().cuda()

# Dummy input data
coordinates = torch.randn(10, 3).cuda()
H = torch.randn(1584, 15252, dtype=torch.complex64).cuda()  # Example shape

# Forward pass
field_estimate, normals = model(coordinates)

# Calculate measurements using forward model
g_pred = forward_model(field_estimate, H)
```

### Normal Vector Calculation

Normal vectors are calculated in the `InstantNGPFieldRepresentation` model as follows:

```python
normals_out = -torch.autograd.grad(torch.sum(out.abs()), coordinates, create_graph=True)[0]
normals_out = safe_normalize(normals_out).float()
normals_out[torch.isnan(normals_out)] = 0
```

These normal vectors are **not currently used in the optimization** but can be applied to model **Lambertian scattering** or compute **transmission probabilities** based on predicted scattering values.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Developed with InstantNGP and PyTorch for radar data processing.