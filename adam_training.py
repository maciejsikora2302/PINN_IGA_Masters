from B_Splines import B_Splines
import numpy as np
import torch
from torch.functional import F
import scipy.interpolate as spi

bs = B_Splines(torch.linspace(0,1,100), 2)



def traing_ADAM(x: torch.Tensor, bs: B_Splines, loss_function: ):
    losses = []
    
    opt = torch.optim.Adam(bs.parameters(), lr=0.001)
    dtype = torch.float
    for t in range(10000):
        y_pred = bs.calculate_BSpline_1D(x)
        loss = F.mse_loss(y_pred, y).sqrt()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss)
        print(t, loss.item())
        
        if t % 100 == 99:
            print(t, loss.item())


# x = torch.Tensor(torch.linspace(0, 1, 10))

# GT data
# print(bs.calculate_BSpline_1D(x))
# print(spi.splev(x.detach(), (torch.linspace(0,1,100).detach(), bs.coefs.detach(), 2)))




# y = y_pred.sum()
# y.backward()
# print(bs.coefs.grad)
# y = y_pred.sum()
# print(bs.coefs)
# y.backward()
# print(bs.coefs.grad)
# print(x.grad)

