# InstantNGP Field Representation for Radar Data Processing

This repository implements an Instant Neural Graphics Primitives (InstantNGP) model using `tiny-cuda-nn` for predicting field estimates over a 3D grid. The model computes measurements using a forward model with complex values, designed for applications in radar data processing. The model includes a neural representation using multi-resolution hash encoding and computes field estimates based on a learned function.

## Mathematical Formulation

### Field Representation

The neural field representation maps 3D coordinates to complex field values:

$F_θ: \mathbb{R}^3 \rightarrow \mathbb{C}$

where $F_θ$ is parameterized by the neural network weights θ. For a given point $\mathbf{x} = (x, y, z)$, the field value is:

$F_θ(\mathbf{x}) = \text{MLP}(γ(\mathbf{x}))$

where $γ(\mathbf{x})$ is the multi-resolution hash encoding.

### Forward Model

The relationship between field estimates $\mathbf{f}$ and measurements $\mathbf{g}$ is given by:

$\mathbf{f} = -\mathbf{H}^H\mathbf{g}$

Therefore, the measurements are computed using the pseudoinverse:

$\mathbf{g} = -(\mathbf{H}^H)^+\mathbf{f}$

where $\mathbf{H}^H$ is the conjugate transpose of $\mathbf{H}$, and $(\cdot)^+$ denotes the pseudoinverse operation.

### Normal Vector Calculation

Normal vectors are computed as the normalized gradient of the field magnitude:

$\mathbf{n}(\mathbf{x}) = -\frac{\nabla |F_θ(\mathbf{x})|}{\|\nabla |F_θ(\mathbf{x})|\|}$

## Features

- **Multi-resolution Hash Encoding**: The model utilizes efficient multi-level hash encoding for high-resolution spatial representations.
- **Complex-Valued Forward Model**: The model computes complex field estimates and includes a forward model to predict measurements.
- **Normal Vector Calculation**: The model calculates normal vectors at each point in the 3D grid, though these are not currently used in optimization.
- **SIREN Architecture**: Implementation follows the SIREN (Sinusoidal Representation Networks) approach using sine activation functions.
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

## Implementation

The model implementation follows the SIREN (Sinusoidal Representation Networks) architecture, which uses sine activation functions to learn continuous signals. The choice of activation function determines the MLP type:

- With sine activation: Uses `CutlassMLP` (current implementation)
- With ReLU activation: Uses `FullyFusedMLP`

This implementation closely follows the SIREN paper's approach of using intermediate sine activations for better representation of complex signals.

### Model Implementation

```python
import torch
import tinycudann as tcnn

class InstantNGPFieldRepresentation(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = get_config()

        encoding_config = config['encoding']
        network_config = config['MLP']

        self.encoding = tcnn.Encoding(
            n_input_dims=config['model']['input_dim'],
            encoding_config=encoding_config,
        )
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=config['model']['output_dim'],
            network_config=network_config,
        )
    
    def forward(self, coordinates):
        with torch.enable_grad():
            coordinates.requires_grad_(True)
            if not coordinates.is_cuda:
                coordinates = coordinates.to('cuda')

            x_encoded = self.encoding(coordinates)
            out = self.network(x_encoded).float()

            normals_out = - torch.autograd.grad(torch.sum(out.abs()),
                                              coordinates, create_graph=True)[0]
            normals_out = safe_normalize(normals_out).float()
            normals_out[torch.isnan(normals_out)] = 0
            
        return torch.complex(real=out[:, 0], imag=out[:, 1]).to('cuda'), normals_out
```

### Forward Model

```python
def forward_model(f_est, H):
    ''' Forward model to compute the measurements
        field_estimate = -H.conj().T @ g 
        Therefore, g = -H_pseudo_inv @ field_estimate'''
    H_pseudo_inv = torch.linalg.pinv(H.conj().T)
    g = -H_pseudo_inv @ f_est
    return g
```

### Configuration and MLP Architecture

```python
def get_config():
    config = {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "MLP": {
            # Using CutlassMLP for sine activation (SIREN architecture)
            # Change to "FullyFusedMLP" if using ReLU activation
            "otype": "CutlassMLP",
            "activation": "Sine",  # Change to "ReLU" if using FullyFusedMLP
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 2
        },
        "model": {
            "input_dim": 3, 
            "output_dim": 2
        }
    }
    return config
```

The MLP architecture follows these key principles:

1. **SIREN Implementation**:
   - Uses sine activation functions for all intermediate layers
   - Follows the periodic activation pattern from the SIREN paper
   - Better suited for representing continuous signals and their derivatives

2. **MLP Type Selection**:
   ```python
   "MLP": {
       # For SIREN architecture:
       "otype": "CutlassMLP",
       "activation": "Sine",
       
       # For ReLU-based architecture:
       # "otype": "FullyFusedMLP",
       # "activation": "ReLU",
   }
   ```

### Utility Functions

```python
def safe_normalize(x, eps=1e-4):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
```

## Multi-resolution Hash Encoding

The hash encoding function $γ$ maps input coordinates to a feature vector using multiple resolution levels:

$γ(\mathbf{x}) = \text{concat}_{l=0}^{L-1}(E_l(\mathbf{x}))$

where $E_l$ is the encoding at level $l$, and $L$ is the total number of levels. Each level operates at a different resolution, allowing the model to capture both fine and coarse details.

The current implementation uses:
- 16 levels of resolution
- 2 features per level
- Base resolution of 16
- Scale factor of 1.5 between levels

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

## Output Details

The model outputs:
1. Field estimates: Complex-valued tensor representing the field at each input coordinate
2. Normal vectors: 3D vectors representing the gradient direction at each point

The forward model then converts these field estimates into measurements using the pseudo-inverse of the conjugate transpose of the sensing matrix.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Omkar Vengurlekar: ovengurl@asu.edu

## References

1. Sitzmann, V., Martel, J. N., Bergman, A. W., Lindell, D. B., Wetzstein, G., & SIREN: Implicit Neural Representations with Periodic Activation Functions.
2. Müller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander, & InstantNGP: Instant neural graphics primitives with a multiresolution hash encoding.
```