from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f, f_spline, dfdx_spline
import numpy as np
from B_Splines import B_Splines
from general_parameters import general_parameters, logger
from typing import Callable
import math

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(math.pi*x).reshape(-1,1)
    # res = x.reshape(-1,1)
    # res = torch.zeros_like(x).reshape(-1,1)
    return res

def precalculations_2D(x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, colocation: bool = False):
    eps_interior = general_parameters.eps_interior
    degree = general_parameters.spline_degree
    knot_vector_length = general_parameters.knot_vector_length
    coefs_vector_length = general_parameters.coefs_vector_length

    linspace = torch.linspace(0, 1, knot_vector_length)
    coefs = torch.ones(coefs_vector_length)
    sp = B_Splines(linspace, degree, coefs=coefs, dims=2) if sp is None else sp

    coefs_int = torch.Tensor(np.random.randint(0, 2, (coefs_vector_length, )))
    sp_coloc = B_Splines(linspace, degree, coefs=coefs_int, dims=2)

    v = sp.calculate_BSpline_2D(x, t)
    v_coloc = sp_coloc.calculate_BSpline_2D(x, t)

    if not colocation:
        return eps_interior, sp, sp.degree, sp.coefs, v
    else:
        return eps_interior, sp, sp.degree, sp.coefs, v_coloc
    
    

def precalculations_1D(x:torch.Tensor, sp: B_Splines = None, colocation: bool = False):
    eps_interior = general_parameters.eps_interior
    knot_vector_length = general_parameters.knot_vector_length
    degree = general_parameters.spline_degree
    coefs_vector_length = general_parameters.coefs_vector_length

    linspace = torch.linspace(0, 1, knot_vector_length)
    coefs = torch.ones(coefs_vector_length)
    #coefs random floats between 0 and 1 as a tensor
    coefs = torch.Tensor(np.random.rand(coefs_vector_length))
    sp = B_Splines(linspace, degree, coefs=coefs) if sp is None else sp

    coefs_int = torch.Tensor(np.random.randint(0, 2, (coefs_vector_length, )))
    sp_coloc = B_Splines(linspace, degree, coefs=coefs_int)

    v = sp.calculate_BSpline_1D(x)
    v_coloc = sp_coloc.calculate_BSpline_1D(x)
    
    if not colocation:
        return eps_interior, sp, sp.degree, sp.coefs, v
    else:
        return eps_interior, sp, sp.degree, sp.coefs, v_coloc


def interior_loss_weak(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)
    
    if dims == 1:
        x = x.cuda()

    if dims == 1:
        
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)
        v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()

        tensor_to_integrate = dfdx(pinn, x, t, order=1).cuda() * v \
            + eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x
        n = x.shape[0]
        loss = torch.trapezoid(tensor_to_integrate, dx = 1/n)
        # tensor_to_integrate = dfdx(pinn, x, t, order=1).cuda() \
        #     + eps_interior*dfdx(pinn, x, t, order=1)
        # n = x.shape[0]
        # loss = torch.trapezoid(tensor_to_integrate, dx = 1/n)


        #print all components of loss_weak
        logger.debug("Loss weak components:")

        logger.debug(f"dfdx(pinn, x, t, order=1).cuda(): {dfdx(pinn, x, t, order=1).cuda()}")
        logger.debug(f"v: {v}")
        logger.debug(f"dfdx(pinn, x, t, order=1).cuda() * v: {dfdx(pinn, x, t, order=1).cuda() * v}")
        logger.debug(f"eps_interior*dfdx(pinn, x, t, order=1): {eps_interior*dfdx(pinn, x, t, order=1)}")
        logger.debug(f"v_deriv_x: {v_deriv_x}")
        logger.debug(f"eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x: {eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x}")
        logger.debug(f"Loss weak: {loss}")

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_dx(x, t) #.cuda()
        v_deriv_t = sp.calculate_BSpline_2D_deriv_dt(x, t) # .cuda()

        n_x = x.shape[0]
        n_t = t.shape[0]

        loss = torch.trapezoid(torch.trapezoid(
            
            # dfdt(pinn, x, t, order=1).cuda() * v
            # + eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x
            # + eps_interior*dfdt(pinn, x, t, order=1) * v_deriv_t
            dfdt(pinn, x, t, order=1).cpu() * v.cpu()
            + eps_interior*dfdx(pinn, x, t, order=1).cpu() * v_deriv_x.cpu()
            + eps_interior*dfdt(pinn, x, t, order=1).cpu() * v_deriv_t.cpu()
            
            , dx = 1/n_x), dx = 1/n_t)
        
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

def interior_loss_colocation(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2):

    if dims == 1:
        x = x.cuda()

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp, colocation = True)

        loss = (dfdx(pinn, x, order=1) - eps_interior*dfdx(pinn, x, order=2)) * v

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp, colocation=True)

        loss = (dfdt(pinn, x, t, order=1).cpu() - 
                eps_interior*dfdt(pinn, x, t, order=2).cpu()
                -eps_interior*dfdx(pinn, x, t, order=2).cpu()) * v.cpu()

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

        
def interior_loss_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 1):

    if dims == 1:
        x = x.cuda()

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)
        tensor_to_integrate = (
            - eps_interior*dfdx(pinn, x, order=2)
            + dfdx(pinn, x, order=1) 
            ) * v
        
        n = x.shape[0]
        loss = torch.trapezoid(tensor_to_integrate, dx = 1/n)

    elif dims == 2:

        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)
        n_x = x.shape[0]
        n_t = t.shape[0]

        loss = torch.trapezoid(torch.trapezoid(

            (dfdt(pinn, x, t, order=1).cpu() 
                            - eps_interior*dfdt(pinn, x, t, order=2).cpu()
                            - eps_interior*dfdx(pinn, x, t, order=2).cpu()) * v.cpu()

                            , dx=1/n_x), dx=1/n_t)

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()


def interior_loss_weak_and_strong(pinn: PINN, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2):

    if dims == 1:
        x = x.cuda()

    if dims == 1:

        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()
        n = x.shape[0]

        # print(v.shape, v_deriv_x.shape)
        # print(x.shape)
        
        # #print all components of loss_weak
        # logger.debug("loss_weak components:")
        # logger.debug("dfdx(pinn, x, order=1).cuda() * v")
        # logger.debug(dfdx(pinn, x, order=1).cuda() * v)
        # logger.debug("eps_interior*dfdx(pinn, x, order=2) * v_deriv_x")
        # logger.debug(eps_interior*dfdx(pinn, x, order=2) * v_deriv_x)
        # logger.debug("+"*100)
        # logger.debug("eps_interior")
        # logger.debug(eps_interior)
        # logger.debug("dfdx(pinn, x, order=2)")
        # logger.debug(dfdx(pinn, x, order=2))
        # logger.debug("v_deriv_x")
        # logger.debug(v_deriv_x)
        # logger.debug("+"*100)
        # logger.debug("final: torch.trapezoid(dfdx(pinn, x, order=1).cuda() * v + eps_interior*dfdx(pinn, x, order=2) * v_deriv_x, dx=1/n)")
        # logger.debug(torch.trapezoid(dfdx(pinn, x, order=1).cuda() * v + eps_interior*dfdx(pinn, x, order=1) * v_deriv_x, dx=1/n))
        # logger.debug("loss_strong components:")
        # logger.debug("eps_interior*dfdx(pinn, x, order=2)")
        # logger.debug(eps_interior*dfdx(pinn, x, order=2))
        # logger.debug("dfdx(pinn, x, order=1)")
        # logger.debug(dfdx(pinn, x, order=1))
        # logger.debug("final: torch.trapezoid((- eps_interior*dfdx(pinn, x, order=2) + dfdx(pinn, x, order=1)) * v, dx=1/n)")
        # logger.debug(torch.trapezoid((- eps_interior*dfdx(pinn, x, order=2) + dfdx(pinn, x, order=1)) * v, dx=1/n))


        loss_weak = torch.trapezoid(
            dfdx(pinn, x, order=1).cuda() * v
            + eps_interior*dfdx(pinn, x, order=1) * v_deriv_x, dx=1/n
            )

        loss_strong = torch.trapezoid((
            - eps_interior*dfdx(pinn, x, order=1)
            + dfdx(pinn, x, order=2) 
            ) * v, dx=1/n)
        

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_dx(x, t)
        v_deriv_t = sp.calculate_BSpline_2D_deriv_dt(x, t)
        
        n_x = x.shape[0]
        n_t = t.shape[0]

        loss_weak = torch.trapezoid(torch.trapezoid(
            
            dfdt(pinn, x, t, order=1).cpu() * v.cpu()
            + eps_interior*dfdx(pinn, x, t, order=1).cpu() * v_deriv_x.cpu()
            + eps_interior*dfdt(pinn, x, t, order=1).cpu() * v_deriv_t.cpu()

            , dx = 1/n_x), dx = 1/n_t)
        
        loss_strong = torch.trapezoid(torch.trapezoid(

            (dfdt(pinn, x, t, order=1).cpu() 
                            - eps_interior*dfdt(pinn, x, t, order=2).cpu()
                            - eps_interior*dfdx(pinn, x, t, order=2).cpu()) * v.cpu()

                            , dx=1/n_x), dx=1/n_t)
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


    return (loss_weak.pow(2) + loss_strong.pow(2)).mean()


def interior_loss_weak_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2, mode: str = 'Adam'):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)
    
    if dims == 1:
        x = x.cuda()

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)
        v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()

        tensor_to_integrate = spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v \
            + eps_interior*spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v_deriv_x
        n = x.shape[0]
        loss = torch.trapezoid(tensor_to_integrate, dx = 1/n)
    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_dx(x, t) #.cuda()
        v_deriv_t = sp.calculate_BSpline_2D_deriv_dt(x, t) # .cuda()

        n_x = x.shape[0]
        n_t = t.shape[0]

        loss = torch.trapezoid(torch.trapezoid(
            
            spline.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).cpu() * v.cpu()
            + eps_interior*spline.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).cpu() * v_deriv_x.cpu()
            + eps_interior*spline.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).cpu() * v_deriv_t.cpu()
            
            , dx = 1/n_x), dx = 1/n_t)
        
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

def interior_loss_colocation_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2, mode: str = 'Adam'):

    if dims == 1:
        x = x.cuda()

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp, colocation = True)

        loss = (spline.calculate_BSpline_1D_deriv_dx(x, mode=mode) - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode)) * v

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp, colocation=True)

        loss = (spline.calculate_BSpline_2D_deriv_dt(x,t, mode=mode).cpu() - 
                eps_interior*spline.calculate_BSpline_2D_deriv_dtdt(x,t, mode=mode).cpu()
                -eps_interior*spline.calculate_BSpline_2D_deriv_dxdx(x,t, mode=mode).cpu()) * v.cpu()

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

        
def interior_loss_strong_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 1, mode: str = 'Adam'):

    if dims == 1:
        x = x.cuda()

    if dims == 1:
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)
        tensor_to_integrate = (
            - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode)
            + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode)
            ) * v
        
        n = x.shape[0]
        loss = torch.trapezoid(tensor_to_integrate, dx = 1/n)

    elif dims == 2:

        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)
        n_x = x.shape[0]
        n_t = t.shape[0]

        loss = torch.trapezoid(torch.trapezoid(

            (spline.calculate_BSpline_2D_deriv_dt(x,t, mode=mode).cpu() 
                            - eps_interior*spline.calculate_BSpline_2D_deriv_dtdt(x,t, mode=mode).cpu()
                            - eps_interior*spline.calculate_BSpline_2D_deriv_dxdx(x,t, mode=mode).cpu()) * v.cpu()

                            , dx=1/n_x), dx=1/n_t)

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()


def interior_loss_weak_and_strong_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, sp: B_Splines = None, dims: int = 2, mode: str = 'Adam'):

    if dims == 1:
        x = x.cuda()

    if dims == 1:

        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()
        n = x.shape[0]


        loss_weak = torch.trapezoid(
            spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v
            + eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode) * v_deriv_x, dx=1/n
            )

        loss_strong = torch.trapezoid((
            - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode)
            + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode) 
            ) * v, dx=1/n)
        

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp)

        v_deriv_x = sp.calculate_BSpline_2D_deriv_dx(x, t)
        v_deriv_t = sp.calculate_BSpline_2D_deriv_dt(x, t)
        
        n_x = x.shape[0]
        n_t = t.shape[0]

        loss_weak = torch.trapezoid(torch.trapezoid(
            
            spline.calculate_BSpline_2D_deriv_dt(x,t, mode=mode).cpu() * v.cpu()
            + eps_interior*spline.calculate_BSpline_2D_deriv_dx(x,t, mode=mode).cpu() * v_deriv_x.cpu()
            + eps_interior*spline.calculate_BSpline_2D_deriv_dt(x,t, mode=mode).cpu() * v_deriv_t.cpu()

            , dx = 1/n_x), dx = 1/n_t)
        
        loss_strong = torch.trapezoid(torch.trapezoid(

            (spline.calculate_BSpline_2D_deriv_dt(x,t).cpu() 
                            - eps_interior*spline.calculate_BSpline_2D_deriv_dtdt(x,t, mode=mode).cpu()
                            - eps_interior*spline.calculate_BSpline_2D_deriv_dxdx(x,t, mode=mode).cpu()) * v.cpu()

                            , dx=1/n_x), dx=1/n_t)
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")


    return (loss_weak.pow(2) + loss_strong.pow(2)).mean()


def boundary_loss_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor, dims: int = 2):

    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_loss_right = f_spline(spline, x_raw)

         # -eps*u'(0)+u(0)-1.0=0
        # boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_xf = x #first point = 0
        boundary_loss_left  = -general_parameters.eps_interior * dfdx_spline(spline, boundary_xf)[0] + \
            + f_spline(spline, boundary_xf)[0]-1.0

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()
    elif dims == 2:
        t_raw = torch.unique(t).reshape(-1, 1).detach()
        t_raw.requires_grad = True
        
        boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
        boundary_loss_left = f_spline(spline, boundary_left, t_raw)

        boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]
        boundary_loss_right = f_spline(spline, boundary_right, t_raw)

        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_top = torch.ones_like(x_raw, requires_grad=True) * t[-1]
        boundary_loss_top = f(pinn, boundary_top, x_raw)

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean() + boundary_loss_top.pow(2).mean()
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

def boundary_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor = None, dims: int = 2):

    if dims == 1:
        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True
        
        ones = torch.ones_like(x_raw, requires_grad=True) * x_raw[-1]
        boundary_loss_right = f(pinn, ones)

        zeros = torch.ones_like(x_raw, requires_grad=True) * x_raw[0]
        #  -eps*u'(0)+u(0)-1.0=0
        #  boundary_xf = x[0].reshape(-1, 1) #first point = 0

        boundary_loss_left  = -general_parameters.eps_interior * dfdx(pinn, zeros) + f(pinn, zeros)-1.0

        # ones = torch.ones_like(x)
        # zeros = torch.zeros_like(x)

        # ones.requires_grad = True
        # zeros.requires_grad = True

        # boundary_loss_right = f(pinn, ones)

        # boundary_loss_left = -general_parameters.eps_interior * dfdx(pinn, zeros) + f(pinn, zeros) - ones
        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()
    
    elif dims == 2:
        t_raw = torch.unique(t).reshape(-1, 1).detach()
        t_raw.requires_grad = True
        
        boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
        print(boundary_left)
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
    #print all weights
    # print("weight_f: ", weight_f)
    # print("weight_b: ", weight_b)
    # print("weight_i: ", weight_i)

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
            weight_f * interior_loss_function(spline, x, t, dims=dims) + \
            weight_i * initial_loss_spline(spline, x, t, dims=dims)
        
        final_loss += weight_b * boundary_loss_spline(spline, x, t, dims=dims)

        return final_loss if not verbose else (final_loss, interior_loss_function(spline, x, t), initial_loss_spline(spline, x, t, dims=dims), boundary_loss_spline(spline, x, t, dims=dims))

    elif dims == 2:
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")