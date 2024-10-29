import torch
import tinycudann as tcnn

def forward_model(f_est, H):
    ''' Forward model to compute the measurements
        field_estimate = -H.conj().T @ g 
        Therefore, g = -H @ field_estimate'''
    H_pseudo_inv = torch.linalg.pinv(H.conj().T)
    g = -H_pseudo_inv @ f_est
    return g

def safe_normalize(x, eps=1e-4):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_config():

    config = {
        "encoding":{
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5},

        "MLP":{
            "otype": "CutlassMLP",
            "activation": "Sine", 
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 2},

        "model":{
            "input_dim":3, 
            "output_dim":2}
    }
    return config

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