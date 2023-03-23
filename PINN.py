import torch
import torch.nn as nn
from scipy.interpolate import BSpline
from general_parameters import general_parameters

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False, dims: int = 1):

        super().__init__()

        self.pinning = pinning
        self.dims = dims
        # self.layer_in = nn.Linear(2, dim_hidden)
        # self.layer_out = nn.Linear(dim_hidden, 1)
        if dims == 1:
            self.layer_in = nn.Linear(1, dim_hidden)
            self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        if self.dims == 1:

            x_stack = torch.cat([x], dim=1)        
            out = self.act(self.layer_in(x_stack))
            for layer in self.middle_layers:
                out = self.act(layer(out))
            logits = self.layer_out(out)

            # if requested pin the boundary conditions 
            # using a surrogate model: (x - 0) * (x - L) * NN(x)
            if self.pinning:
                logits *= (x - x[0]) * (x - x[-1])
            
            return logits


        else:

            x_stack = torch.cat([x, t], dim=1)        
            out = self.act(self.layer_in(x_stack))
            for layer in self.middle_layers:
                out = self.act(layer(out))
            logits = self.layer_out(out)

        # if requested pin the boundary conditions 
        # using a surrogate model: (x - 0) * (x - L) * NN(x)
        if self.pinning:
            logits *= (x - x[0]) * (x - x[-1])
        
        return logits