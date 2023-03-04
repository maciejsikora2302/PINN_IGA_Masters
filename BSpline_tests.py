from B_Splines import B_Splines
from general_parameters import general_parameters
import numpy as np
import torch

bs = B_Splines(torch.Tensor(np.linspace(0,1,100)), 3)
x = torch.Tensor([0.59837, 0.59837])
t = torch.Tensor([0.59837, 0.59837])


y_pred = bs.calculate_BSpline_1D_deriv_dx(x)
# y = y_pred
# y.backward()
print(y_pred)
# y = y_pred.sum()
# print(bs.coefs)
# y.backward()
# print(bs.coefs.grad)
# print(x.grad)

