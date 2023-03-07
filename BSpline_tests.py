from B_Splines import B_Splines
import numpy as np
import torch
from torch.functional import F
import scipy.interpolate as spi


bs = B_Splines(torch.linspace(0,1,1000), 2)
opt = torch.optim.Adam(bs.parameters(), lr=0.001)

dtype = torch.float
device = torch.device("cuda")
x = torch.Tensor([0.33, 0.5, 0.774, 0.97])
t = torch.Tensor([0.33, 0.5, 0.774, 0.97])
print(bs.calculate_BSpline_2D(x, t))
# GT data
y = torch.sin(x)
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

