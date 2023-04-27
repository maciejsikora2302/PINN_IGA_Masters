import torch
import torch.nn as nn
import json

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False, dims: int = 1, dim_layer_in = 1, dim_layer_out: int = 1, pinn_learns_coeff: bool = False):

        super().__init__()

        self.pinning = pinning
        self.dims = dims
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dim_layer_in = dim_layer_in
        self.dim_layer_out = dim_layer_out
        self.pinn_learns_coeff = pinn_learns_coeff

        self.layer_in = nn.Linear(dims, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, dim_layer_out)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )

        self.act = act

    def forward(self, x, t):
        
        x_stack = torch.cat([x], dim=1) if self.dims == 1 else torch.cat([x, t], dim=1)

        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        if self.pinning:
            logits *= (x - x[0]) * (x - x[-1])
        
        return logits
    

    def __str__(self) -> str:
        json_repr = {
            "num_hidden": self.num_hidden,
            "dim_hidden": self.dim_hidden,
            "act": str(self.act),
            "pinning": self.pinning,
            "dims": self.dims,
            "dim_layer_in": self.dim_layer_in,
            "dim_layer_out": self.dim_layer_out,
            "pinn_learns_coeff": self.pinn_learns_coeff,
        }
        return json.dumps(json_repr)


def get_activation_function(activation_name=None, activation_function=None):
    activation_mapping = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "Softmax": nn.Softmax,
        "Softplus": nn.Softplus,
        "Softsign": nn.Softsign,
        "TanhShrink": nn.Tanhshrink,
        "Softmin": nn.Softmin,
        "SELU": nn.SELU,
        "GELU": nn.GELU,
        "LogSigmoid": nn.LogSigmoid,
    }

    reverse_activation_mapping = {v: k for k, v in activation_mapping.items()}

    if activation_name:
        activation_name = activation_name.replace("(", "")
        activation_name = activation_name.replace(")", "")
        # print(activation_name)
        # print(activation_mapping.get(activation_name, None))
        return activation_mapping.get(activation_name, None)
    elif activation_function:
        # print(activation_function)
        # print(reverse_activation_mapping.get(activation_function, None))
        return reverse_activation_mapping.get(activation_function, None)
    else:
        raise ValueError("Either activation_name or activation_function must be provided.")
