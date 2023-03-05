from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f, f_spline, dfdx_spline
import numpy as np
from B_Splines import B_Splines
from general_parameters import general_parameters
from typing import Callable
import math

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(math.pi*x).reshape(-1,1)
    return res

def precalculations_2D(x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None):
    eps_interior = general_parameters.eps_interior

    degree, knot_vector_length = general_parameters.spline_degree, general_parameters.knot_vector_length
    linspace = torch.linspace(0, 1, int(knot_vector_length))
    sp = B_Splines(linspace, degree, dims=2) if sp is None else sp
    v = sp.calculate_BSpline_2D(x, t)
    
    return eps_interior, sp, sp.degree, sp.coefs, v

def precalculations_1D(x:torch.Tensor, sp: B_Splines = None, colocation: bool = False):
    eps_interior = general_parameters.eps_interior
    knot_vector_length = general_parameters.knot_vector_length
    degree = general_parameters.spline_degree

    linspace = torch.linspace(0, 1, int(knot_vector_length)) if sp is None else sp
    sp = B_Splines(linspace, degree)

    coef_int = torch.Tensor(np.random.randint(0, 2, (len(sp.knot_vector), )))
    # sp_coloc = B_Splines(linspace, degree, coefs=coef_int)

    v = sp.calculate_BSpline_1D(x)
    # v_coloc = sp_coloc.calculate_BSpline_1D(x)

    # print(v_coloc.shape, v.shape)
    
    if not colocation:
        return eps_interior, sp, sp.degree, sp.coefs, v
    else:
        return eps_interior, sp, sp.degree, sp.coefs#, v_coloc


def interior_loss_weak(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2, extended: bool = False):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)
    x = x.cuda()
    t = t.cuda() if dims == 2 else None


    if dims == 1:
        eps_interior, sp, _, coef_float, v = precalculations_1D(x, sp)
        v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()
        loss = torch.trapezoid(
            dfdx(pinn, x, t, order=1).cuda() * v
            + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
            , dx = 0.01)

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_dx(x, t).cuda()
        v_deriv_t = sp.calculate_BSpline_2D_deriv_dt(x, t).cuda()
        loss = torch.trapezoid(torch.trapezoid(
            
            dfdx(pinn, x, t, order=1).cuda() * v
            + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
            + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
            
            , dx = 0.01), dx = 0.01)
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

def interior_loss_colocation(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, extended: bool = False, dims: int = 2):

    x = x.cuda()
    t = t.cuda() if dims == 2 else None

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp, colocation = True)

        loss = (dfdx(pinn, x, order=1) - eps_interior*dfdx(pinn, x, order=2)) * v

    elif dims == 2:
        eps_interior, sp, degree_1, degree_2, _, _, _ = precalculations_2D(x, t, sp)
        coef1 = np.random.randint(0, 2, len(sp.knot_vector))
        coef2 = np.random.randint(0, 2, len(sp.knot_vector))

        v = sp.calculate_BSpline_2D(x.detach(), t.detach(), degree_1, degree_2, coef1, coef2)
        loss = (dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)) * v

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss if extended else loss.pow(2).mean()

        
def interior_loss_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, extended: bool = False, dims: int = 2):

    x = x.cuda()
    t = t.cuda() if dims == 2 else None


    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)


        loss = torch.trapezoid((
            - eps_interior*dfdx(pinn, x, order=2)
            + dfdx(pinn, x, order=1) 
            )) * v

    elif dims == 2:

        eps_interior, sp, _, _, _, _, v = precalculations_2D(x, t, sp)

        loss = torch.trapezoid(torch.trapezoid((
            dfdt(pinn, x, t, order=1) 
            - eps_interior*dfdt(pinn, x, t, order=2)
            - eps_interior*dfdx(pinn, x, t, order=2)))) * v

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss if extended else loss.pow(2).mean()


def interior_loss_weak_and_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2):

    x = x.cuda()
    t = t.cuda() if dims == 2 else None


    if dims == 1:

        eps_interior, sp, _, _, v = precalculations_1D(x, sp)


        loss_weak = (dfdx(pinn, x, order=1) - eps_interior*dfdx(pinn, x, order=2)) * v

        loss_strong = torch.trapezoid((
            - eps_interior*dfdx(pinn, x, order=2)
            + dfdx(pinn, x, order=1) 
            )) * v

    elif dims == 2:
        eps_interior, sp, degree_1, degree_2, coef_float, coef_float_2, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
        v_deriv_t = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
        loss_weak = torch.trapezoid(torch.trapezoid(
            
            dfdx(pinn, x, t, order=1) * v
            + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
            + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
            
            , dx = 0.01), dx = 0.01)
        
        loss_strong = torch.trapezoid(torch.trapezoid((dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)))) * v
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


    return loss_weak.pow(2).mean() + loss_strong.pow(2).mean()

# TODO: Ogarnąć IGA loss function
def iga_loss(sp: B_Splines, x: torch.Tensor, t: torch.Tensor, dims: int = 2): # It's just a classic version, w/o collocation

    x = x.cuda()
    t = t.cuda() if dims == 2 else None

    if dims == 1:

        eps_interior, sp, degree, coef_float, v = precalculations_1D(x, sp)


        spline_deriv_dx = torch.Tensor(sp.calculate_BSpline_1D_deriv(x.detach(), coef_float, degree, order=1))
        spline_deriv_dxdx = torch.Tensor(sp.calculate_BSpline_1D_deriv(x.detach(), coef_float, degree, order=2))

        loss_iga = spline_deriv_dx - eps_interior * spline_deriv_dxdx

    elif dims == 2:
        eps_interior, sp, degree_1, degree_2, coef_float, coef_float_2, _ = precalculations_2D(x, t, sp)

        spline_deriv_dx = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
        spline_deriv_dxdx = sp.calculate_BSpline_2D_deriv_x(spline_deriv_dx, t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=2)
        spline_deriv_dt = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
        spline_deriv_dtdt = sp.calculate_BSpline_2D_deriv_t(x.detach(), spline_deriv_dt, degree_1, degree_2, coef_float, coef_float_2, order=2)

        loss_iga = spline_deriv_dx - eps_interior * (spline_deriv_dxdx + spline_deriv_dtdt)
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss_iga.pow(2).mean()

def iga_loss_deriv(sp: B_Splines, x: torch.Tensor, t: torch.Tensor, dims: int = 2): # It's just a classic version, w/o collocation

    x = x.cuda()
    t = t.cuda() if dims == 2 else None

    if dims == 1:

        eps_interior, sp, degree, coef_float, v = precalculations_1D(x, sp)

        if degree < 3:
            raise ValueError("Degree must be at least 3 to calculate 3rd derivative")


        spline_deriv_dxdx = torch.Tensor(sp.calculate_BSpline_1D_deriv(x.detach(), coef_float, degree, order=2))
        spline_deriv_dxdxdx = torch.Tensor(sp.calculate_BSpline_1D_deriv(x.detach(), coef_float, degree, order=3))

        loss_iga = spline_deriv_dxdx - eps_interior * spline_deriv_dxdxdx

    elif dims == 2:
        #TODO implement later
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss_iga.pow(2).mean()

def boundary_loss_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, dims: int = 2):

    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_loss_right = f_spline(spline, x_raw)

         # -eps*u'(0)+u(0)-1.0=0
        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_left  = -general_parameters.eps_interior * dfdx_spline(spline, boundary_xf) + f_spline(spline, boundary_xf)-1.0

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()
    elif dims == 2:
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

def boundary_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor, dims: int = 2):

    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_loss_right = f(pinn, x_raw)

         # -eps*u'(0)+u(0)-1.0=0
        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_left  = -general_parameters.eps_interior * dfdx(pinn, boundary_xf) + f(pinn, boundary_xf)-1.0

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()
    elif dims == 2:
        t_raw = torch.unique(t).reshape(-1, 1).detach()
        t_raw.requires_grad = True
        
        boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
        boundary_loss_left = f(pinn, boundary_left, t_raw)

        boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]
        boundary_loss_right = f(pinn, boundary_right, t_raw)

        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_top = torch.ones_like(x_raw, requires_grad=True) * t[-1]
        boundary_loss_top = f(pinn, boundary_top, x_raw)

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean() + boundary_loss_top.pow(2).mean()
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


def tmp_func(): # function copied from 1D example
    def compute_loss(
        nn_approximator: PINN, x: torch.Tensor = None, device = "cpu"
    ) -> torch.float:
        """Compute the full loss function as interior loss + boundary loss
        This custom loss function is fully defined with differentiable tensors therefore
        the .backward() method can be applied to it
        """
        epsilon = 0.1 #<-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

    # PDE residual -epsilon*u''(x)+u'(x)=0
        interior_loss = - epsilon * dfdx(nn_approximator, x, order=2) + dfdx(nn_approximator, x, order=1)
    
    # u(1)=0
        boundary_xi = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_right = f(nn_approximator, boundary_xi)
        
    # -eps*u'(0)+u(0)-1.0=0
        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_left  = -epsilon * dfdx(nn_approximator, boundary_xf) + f(nn_approximator, boundary_xf)-1.0
        
        # obtain the final MSE loss by averaging each loss term and summing them up
        final_loss = \
            interior_loss.pow(2).mean() + \
            boundary_loss_left.pow(2).mean() + \
            boundary_loss_right.pow(2).mean() 
    
        return final_loss
    pass


def initial_loss_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor = None, dims: int = 2):
    # initial condition loss on both the function and its
    # time first-order derivative
    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        f_initial = initial_condition(x_raw)

        initial_loss_f = f_spline(spline, x_raw) - f_initial 
        initial_loss_df = dfdx_spline(spline, x_raw, order=1)
        return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()
    elif dims == 2:
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


def initial_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor = None, dims: int = 2):
    # initial condition loss on both the function and its
    # time first-order derivative
    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        f_initial = initial_condition(x_raw)

        initial_loss_f = f(pinn, x_raw) - f_initial 
        initial_loss_df = dfdx(pinn, x_raw, order=1)
        return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()
    elif dims == 2:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        f_initial = initial_condition(x_raw)
        t_initial = torch.zeros_like(x_raw)
        t_initial.requires_grad = True

        initial_loss_f = f(pinn, x_raw, t_initial) - f_initial 
        initial_loss_df = dfdt(pinn, x_raw, t_initial, order=1)
        return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


def compute_loss(
    pinn: PINN, x: torch.Tensor = None, t: torch.Tensor = None, 
    weight_f = 1.0, weight_b = 1.0, weight_i = 1.0, 
    verbose = False, interior_loss_function: Callable = interior_loss_weak_and_strong,
    dims: int = 2
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """


    if dims == 1:
        t = None
        final_loss = \
            weight_f * interior_loss_function(pinn, x, t, dims=dims) + \
            weight_i * initial_loss(pinn, x, t, dims=dims)
        
        if not pinn.pinning:
            final_loss += weight_b * boundary_loss(pinn, x, t, dims=dims)

        return final_loss if not verbose else (final_loss, interior_loss_function(pinn, x, t), initial_loss(pinn, x, t, dims=dims), boundary_loss(pinn, x, t, dims=dims))

    elif dims == 2:
        final_loss = \
            weight_f * interior_loss_function(pinn, x, t, dims=dims) + \
            weight_i * initial_loss(pinn, x, t, dims=dims)
        
        if not pinn.pinning:
            final_loss += weight_b * boundary_loss(pinn, x, t, dims=dims)


        return final_loss if not verbose else (final_loss, interior_loss_function(pinn, x, t), initial_loss(pinn, x, t, dims=dims), boundary_loss(pinn, x, t, dims=dims))
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


def compute_loss_spline(
    spline: B_Splines, x: torch.Tensor = None, t: torch.Tensor = None, 
    weight_f = 1.0, weight_b = 1.0, weight_i = 1.0, 
    verbose = False, interior_loss_function: Callable = iga_loss,
    dims: int = 2
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """


    if dims == 1:
        t = None
        final_loss = \
            weight_f * interior_loss_function(spline, x, t, dims=dims) + \
            weight_i * initial_loss_spline(spline, x, t, dims=dims)
        
        final_loss += weight_b * boundary_loss_spline(spline, x, t, dims=dims)

        return final_loss if not verbose else (final_loss, interior_loss_function(spline, x, t), initial_loss_spline(spline, x, t, dims=dims), boundary_loss_spline(spline, x, t, dims=dims))

    elif dims == 2:
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")