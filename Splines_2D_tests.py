from general_parameters import GeneralParameters
from B_Splines import B_Splines
import torch


parameters = GeneralParameters(None)
parameters.pinn_is_solution = True
parameters.n_points_x = 10
parameters.spline_degree = 2
parameters.optimize_test_function = True
parameters.precalculate()

knot_vector = parameters.knot_vector

bs = B_Splines(
    knot_vector=knot_vector,
    degree=3,
    dims=1
)
x=torch.linspace(0, 1, 5)
t=torch.linspace(0, 1, 5)
# x = torch.Tensor([0.03, 0.4, 0.5, 0.7, 0.93])

print(bs.calculate_BSpline_1D_deriv_dxdx(x, mode='NN'))
print(bs.calculate_BSpline_1D_deriv_dxdx(x, mode='Adam'))
# print(bs._B(torch.Tensor([0]), 3, 0, knot_vector))