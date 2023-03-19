import torch
import torch.nn as nn


class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False, input_layer_dims: int = 1, output_layer_dims: int = 1, dims: int = 1):

        super().__init__()

        self.pinning = pinning
        self.dims = dims
        # self.layer_in = nn.Linear(2, dim_hidden)
        # self.layer_out = nn.Linear(dim_hidden, 1)
        if dims == 1:
            self.layer_in = nn.Linear(input_layer_dims, dim_hidden)
            self.layer_out = nn.Linear(dim_hidden, output_layer_dims)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        if self.dims == 1:

            x_stack = torch.transpose(x, 0, 1)
            # x_stack = torch.cat([x, t], dim=1)   
            # print(f"x_stack is {x_stack} and shape {x_stack.shape}")     
            out = self.act(self.layer_in(x_stack))
            for layer in self.middle_layers:
                out = self.act(layer(out))
            logits = self.layer_out(out)
            logits = torch.transpose(logits, 0, 1)

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