import torch
import torch.nn as nn
from scipy.interpolate import BSpline
from general_parameters import general_parameters

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
            out = self.act(self.layer_in(x_stack))
            for layer in self.middle_layers:
                out = self.act(layer(out))
            logits = self.layer_out(out)

            # logits_tmp = torch.clone(logits).cpu().detach().numpy()
            # spl = BSpline(general_parameters.knot_vector, logits_tmp, general_parameters.spline_degree)
            # tmp_x = torch.clone(x).cpu().detach().numpy()
            # f_value = torch.Tensor(spl(tmp_x, 0)).squeeze(dim=1)

            logits = torch.transpose(logits, 0, 1)
            # x_stack = torch.transpose(x, 0, 1)
            # print("x_stack shape:", x_stack.shape)
            # out = self.act(self.layer_in(x_stack))
            # print("out shape after layer_in:", out.shape)
            # for layer in self.middle_layers:
            #     out = self.act(layer(out))
            #     print("out shape after middle layer:", out.shape)
            # logits = self.layer_out(out)
            # print("logits shape after layer_out:", logits.shape)

            # logits_tmp = torch.clone(logits)
            # logits_tmp = torch.transpose(logits_tmp, 0, 1).cpu().detach().numpy()
            # print("logits_tmp shape:", logits_tmp.shape)
            # print("knot_vector shape:", general_parameters.knot_vector.shape)
            # print("logits_tmp shape:", logits_tmp.shape)
            # print("spline degree:", general_parameters.spline_degree)
            # print("coeffs length:", len(logits_tmp))
            # print("knot_vector length:", len(general_parameters.knot_vector))
            # print("and it should be:", general_parameters.knot_vector_length)
            # # print("spline parameters:", general_parameters.knot_vector, logits_tmp, general_parameters.spline_degree)
            # spl = BSpline(general_parameters.knot_vector, logits_tmp, general_parameters.spline_degree)
            # tmp_x = torch.clone(x).cpu().detach().numpy()
            # print("tmp_x shape:", tmp_x.shape)
            # f_value = torch.Tensor(spl(tmp_x, 0)).squeeze(dim=1)
            # print("f_value shape:", f_value.shape)

            
            # # logits = torch.transpose(f_value, 0, 1)
            # # print("logits shape after transpose:", logits.shape)
            # logits = f_value


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