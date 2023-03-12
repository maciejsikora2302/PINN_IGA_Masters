import torch
from PINN import PINN
from B_Splines import B_Splines

def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    # print(f"pinn(x, t) is {pinn(x, t)}")
    # print data sypes of x and t
    return pinn(x.cuda(), t.cuda() if t is not None else torch.zeros_like(x.cuda()).cuda())
    # return pinn(x, t if t is not None else torch.zeros_like(x))

def f_spline(spline: B_Splines, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the spline model"""
    return spline.calculate_BSpline_1D(x)

def dfdx_spline(spline: B_Splines, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate derivative solution from the spline model"""
    return spline.calculate_BSpline_1D_deriv_dx(x)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    # return df_value.cuda()
    return df_value

# def tmp(spline, x, t, coeff):
#     return spline(x, t, coeff)

def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor = None, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)