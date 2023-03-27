import torch
from PINN import PINN
from B_Splines import B_Splines
from general_parameters import general_parameters
import scipy.interpolate as spi
from scipy.interpolate import BSpline


def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    if t is None:
        return pinn(x.cuda(), t)


    # return pinn(x, t if t is not None else torch.zeros_like(x))

def f_spline(spline: B_Splines, x: torch.Tensor, t: torch.Tensor = None, mode: str = 'Adam') -> torch.Tensor:
    """Compute the value of the approximate solution from the spline model"""

    if spline.dims == 1:
        return spline.calculate_BSpline_1D(x, mode=mode).cuda()
    elif spline.dims == 2:
        return spline.calculate_BSpline_2D(x, t, mode=mode)

def dfdx_spline(spline: B_Splines, x: torch.Tensor, t: torch.Tensor = None, mode: str = 'Adam') -> torch.Tensor:
    """Compute the value of the approximate derivative solution from the spline model w.r.t. x"""

    if spline.dims == 1:
        return spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
    elif spline.dims == 2:
        return spline.calculate_BSpline_2D_deriv_dx(x, t, mode=mode)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output

    if not general_parameters.pinn_learns_coeff:
        for _ in range(order):
            df_value = torch.autograd.grad(
                df_value,
                input,
                grad_outputs=torch.ones_like(input),
                create_graph=True,
                retain_graph=True,
            )[0]
    else:
        for _ in range(order):
            df_value = torch.autograd.grad(
                df_value,
                input,
                grad_outputs=torch.ones_like(input).cuda(),
                create_graph=True,
                retain_graph=True,
            )[0]
            #print(df_value.shape)

    # return df_value.cuda()
    return df_value

# def tmp(spline, x, t, coeff):
#     return spline(x, t, coeff)

def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)

def dfdt_spline(spline: B_Splines, x: torch.Tensor, t: torch.Tensor, mode: str = 'Adam'):
    """Derivative of spline with respect to the time variable of the first order"""
    
    return spline.calculate_BSpline_2D_deriv_dt(x, t, mode=mode)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor = None, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)