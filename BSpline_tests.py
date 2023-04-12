from B_Splines import B_Splines
import numpy as np
import torch
from torch.functional import F
import scipy.interpolate as spi
from  general_parameters import general_parameters


# bs = B_Splines(torch.linspace(0,1,1000), 2)
# opt = torch.optim.Adam(bs.parameters(), lr=0.001)

# dtype = torch.float
# device = torch.device("cpu")
# x = torch.Tensor([0.33, 0.5, 0.774, 0.97])
# t = torch.Tensor([0.33, 0.5, 0.774, 0.97])
# print(bs.calculate_BSpline_2D_deriv_dt(x, t))
# GT data
# y = torch.sin(x)
# print(bs.calculate_BSpline_1D(x))
# print(spi.splev(x.detach(), (torch.linspace(0,1,100).detach(), bs.coefs.detach(), 2)))

# losses = []

# for t in range(10000):
#     y_pred = bs.calculate_BSpline_1D(x, mode='adam')
#     loss = F.mse_loss(y_pred, y.cuda()).sqrt()
#     loss.backward()
#     opt.step()
#     opt.zero_grad()
#     losses.append(loss)
#     print(t, loss.item())
    
#     if t % 100 == 99:
#         print(t, loss.item())


# y = y_pred.sum()
# y.backward()
# print(bs.coefs.grad)
# y = y_pred.sum()
# print(bs.coefs)
# y.backward()
# print(bs.coefs.grad)
# print(x.grad)

N_POINTS_X = 15
general_parameters.pinn_is_solution = True
general_parameters.n_points_x = N_POINTS_X
general_parameters.precalculate()

bs = B_Splines(
    knot_vector=general_parameters.knot_vector,
    degree=3,
)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_domain = [0.0, 1.0]
# x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)


# x = x_raw.flatten().reshape(-1, 1).to(device)

# x = torch.linspace(0.0,1.0,10)
# spline_NN = bs.calculate_BSpline_1D(x, mode='NN')
# spline_adam = bs.calculate_BSpline_1D(x, mode='Adam')
# print(spline_NN)
# print(spline_adam)
x = torch.linspace(0,1,10).cuda()
t = torch.linspace(0,1,10).cuda()
# print((bs._get_basis_functions_1D(x).cuda() @ bs.coefs.cuda()).shape)
# print(bs._get_basis_functions_1D(x).shape)

# print(bs.calculate_BSpline_1D_deriv_dx(x, mode='Adam'))
# print(bs.calculate_BSpline_1D_deriv_dxdx(x, mode='NN', order=0))
# print(bs.calculate_BSpline_1D_deriv_dx(x, mode='NN'))

print(bs.calculate_BSpline_1D(x, mode='Adam'))
print(bs.calculate_BSpline_1D(x, mode='NN'))

print(bs.coefs.cuda() @ bs._get_basis_functions_1D(x, order=0).T)