# class DecompositionLinear(nn.Module):
#     def __init__(self, n_features, r):
#         super(DecompositionLinear, self).__init__()
        
#         self.n_features = n_features
#         self.r = r
        
#         self.u = nn.Parameter(torch.empty(n_features, r))
#         self.v = nn.Parameter(torch.empty(r, n_features))

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.n_features)
#         self.u.data.uniform_(-stdv, stdv)
#         self.v.data.uniform_(-stdv, stdv)

#     def forward(self, x):
#         return self.u @ self.v @ x


# https://stackoverflow.com/questions/71332437/is-there-a-pytorch-equivalent-of-tf-custom-gradient
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

# nn.Parameter(torch.empty(n_features, r))

# https://einops.rocks

# https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb

# https://gist.github.com/Carbon225/d6ea4cd9bb4e72b1ea6803f8a322840b


# x = [0, 1/2]
# for i in range(2, 101):
#     xi = x[i-1] + (x[i-1] - x[i-2])/2
#     x.append(xi)
# print(x)

import numpy as np
import torch

def get_unequaly_distribution_points(eps: float = 0.1, density_range: float = .2, n: int = 100, device: torch.device = 'cuda') -> torch.Tensor:
    # Calculating the range of the density function
    eps_prim = 1 - eps
    range_dr_ep = eps_prim - (density_range * eps)

    # Initializing the points as a NumPy array
    points_np = np.zeros(n)

    # Setting the first two points
    points_np[0] = 0.0
    points_np[1] = 0.5
    start_index_for_next_step = 1

    # Calculating the points using the given formula up to range_dr_ep
    for i in range(2, n):
        tmp = points_np[i-1] + (points_np[i-1] - points_np[i-2])/2.0
        if tmp > range_dr_ep:
            start_index_for_next_step = i
            break
        points_np[i] = tmp


    # Equally spreading the remaining points in the range
    linspace = np.linspace(points_np[start_index_for_next_step-1], 1.0, n - start_index_for_next_step + 1)

    points_np[start_index_for_next_step-1:] = linspace

    # Converting the NumPy array to a PyTorch tensor
    points = torch.from_numpy(points_np).to(device)

    return points   

x = get_unequaly_distribution_points(eps = 0.01, density_range = .2, n = 100)
print(x)