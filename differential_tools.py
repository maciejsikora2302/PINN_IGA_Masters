import torch
from PINN import PINN
from B_Splines import B_Splines
from general_parameters import general_parameters
import scipy.interpolate as spi
from scipy.interpolate import BSpline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def f(model, x: torch.Tensor, t: torch.Tensor = None, mode: str = 'Adam') -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    assert isinstance(model, (PINN, B_Splines)), 'The model must be a PINN or B_splines instance'

    if isinstance(model, PINN):
        value = model(x.to(device), t)

    elif isinstance(model, B_Splines):

        if model.dims == 1:
            value = model.calculate_BSpline_1D(x, mode=mode)

        elif model.dims == 2:
            value = model.calculate_BSpline_2D(x, t, mode=mode)

    return value.to(device)


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

    return df_value


def dfdt(model, x: torch.Tensor, t: torch.Tensor, order: int = 1, mode: str = 'Adam'):
    """Derivative with respect to the time variable of arbitrary order"""
    assert isinstance(model, (PINN, B_Splines)), 'The model must be a PINN or B_splines instance'

    if isinstance(model, PINN):
        f_value = f(model, x, t)
        value = df(f_value, t, order=order)
    elif isinstance(model, B_Splines):
        value = model.calculate_BSpline_2D_deriv_dt(x, t, mode=mode)

    return value.to(device)

def dfdx(model, x: torch.Tensor, t: torch.Tensor = None, order: int = 1, mode: str = 'Adam'):
    """Derivative with respect to the spatial variable of arbitrary order"""
    assert isinstance(model, (PINN, B_Splines)), 'The model must be a PINN or B_splines instance'

    if isinstance(model, PINN):
        f_value = f(model, x, t)
        value = df(f_value, x, order=order)
    elif isinstance(model, B_Splines):
        if model.dims == 1:
            value = model.calculate_BSpline_1D_deriv_dx(x, mode=mode)
        elif model.dims == 2:
            value = model.calculate_BSpline_2D_deriv_dx(x, t, mode=mode)
    
    return value.to(device)
