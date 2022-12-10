from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f
import numpy as np
from B_Splines import B_Splines
from general_parameters import general_parameters
from typing import Callable
import math

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(math.pi*x).reshape(-1,1)
    return res

def precalculations(x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):
    eps_interior = general_parameters.eps_interior
    sp = B_Splines(np.linspace(0, 1, int(1/eps_interior * 5))) if sp is None else sp
    degree_1, degree_2 = 2, 2
    coef_float, coef_float_2 = np.random.rand(len(sp.knot_vector)), np.random.rand(len(sp.knot_vector))
    v = sp.calculate_BSpline_2D(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2)

    return eps_interior, sp, degree_1, degree_2, coef_float, coef_float_2, v


def interior_loss_weak(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)

    eps_interior, sp, degree_1, degree_2, coef_float, coef_float_2, v = precalculations(x, t, sp)

    v_deriv_x = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    v_deriv_t = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    loss = torch.trapezoid(torch.trapezoid(
        
        dfdx(pinn, x, t, order=1) * v
        + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
        + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
        
        , dx = 0.01), dx = 0.01)

    return loss.pow(2).mean()

def interior_loss_colocation(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):

    eps_interior, sp, degree_1, degree_2, _, _, _ = precalculations(x, t, sp)
    coef1 = np.random.randint(0, 2, len(sp.knot_vector))
    coef2 = np.random.randint(0, 2, len(sp.knot_vector))

    v = sp.calculate_BSpline_2D(x.detach(), t.detach(), degree_1, degree_2, coef1, coef2)
    loss = (dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)) * v

    return loss.pow(2).mean()

def interior_loss_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):

    eps_interior, sp, _, _, _, _, v = precalculations(x, t, sp)

    loss = torch.trapezoid(torch.trapezoid((dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)))) * v

    return loss.pow(2).mean()


def interior_loss_weak_and_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):

    eps_interior, sp, degree_1, degree_2, coef_float, coef_float_2, v = precalculations(x, t, sp)

    v_deriv_x = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    v_deriv_t = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    loss_weak = torch.trapezoid(torch.trapezoid(
        
        dfdx(pinn, x, t, order=1) * v
        + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
        + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
        
        , dx = 0.01), dx = 0.01)
    
    loss_strong = torch.trapezoid(torch.trapezoid((dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)))) * v


    return loss_weak.pow(2).mean() + loss_strong.pow(2).mean()



def boundary_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor):
    t_raw = torch.unique(t).reshape(-1, 1).detach()
    t_raw.requires_grad = True
    
    boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
    boundary_loss_left = f(pinn, boundary_left, t_raw)

    boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]
    boundary_loss_right = f(pinn, boundary_right, t_raw)

    x_raw = torch.unique(x).reshape(-1, 1).detach()
    x_raw.requires_grad = True

    boundary_top = torch.ones_like(x_raw, requires_grad=True) * t[-1]
    boundary_loss_right = f(pinn, boundary_top, x_raw)


    return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()

def initial_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor):
    # initial condition loss on both the function and its
    # time first-order derivative
    x_raw = torch.unique(x).reshape(-1, 1).detach()
    x_raw.requires_grad = True

    f_initial = initial_condition(x_raw)
    t_initial = torch.zeros_like(x_raw)
    t_initial.requires_grad = True

    initial_loss_f = f(pinn, x_raw, t_initial) - f_initial 
    initial_loss_df = dfdt(pinn, x_raw, t_initial, order=1)

    return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()

def compute_loss(
    pinn: PINN, x: torch.Tensor = None, t: torch.Tensor = None, 
    weight_f = 1.0, weight_b = 1.0, weight_i = 1.0, 
    verbose = False, interior_loss_function: Callable = interior_loss_weak_and_strong
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    final_loss = \
        weight_f * interior_loss_function(pinn, x, t) + \
        weight_i * initial_loss(pinn, x, t)
    
    if not pinn.pinning:
        final_loss += weight_b * boundary_loss(pinn, x, t)

    if not verbose:
        return final_loss
    else:
        return final_loss, interior_loss_function(pinn, x, t), initial_loss(pinn, x, t), boundary_loss(pinn, x, t)
