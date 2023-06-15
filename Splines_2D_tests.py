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
x=torch.linspace(0,1,5)
t=torch.linspace(0,1,5)

print(bs.calculate_BSpline_1zD_deriv_dx(x, mode="Adam"))
print(bs.calculate_BSpline_1D_deriv_dx(x, mode="NN"))
