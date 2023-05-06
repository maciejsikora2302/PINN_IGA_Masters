import torch
import numpy as np

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