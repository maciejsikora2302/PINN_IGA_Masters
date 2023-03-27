from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f, f_spline, dfdx_spline, dfdt_spline
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
    knot_vector_length = len(general_parameters.knot_vector)
    degree = general_parameters.spline_degree
    coefs_vector_length = general_parameters.n_coefs

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

    if general_parameters.pinn_is_solution:
        x = torch.rand_like(x)
        x = torch.sort(x)[0]


    eps_interior = general_parameters.eps_interior
    degree = general_parameters.spline_degree
    coefs_vector_length = general_parameters.n_coefs

    linspace = general_parameters.knot_vector

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


def interior_loss_weak(
        pinn: PINN,
        x:torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        dims: int = 2, 
        test_function: B_Splines = None
        ):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)
    

    if dims == 1:
        
        x = x.cuda()
        eps_interior, sp, _, _, _ = precalculations_1D(x, sp)
        
        if general_parameters.optimize_test_function:
            v = test_function.calculate_BSpline_1D(x, mode="Adam").cuda()
            v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode="Adam").cuda()

            loss = dfdx(pinn, x, t, order=1).cuda() * v \
                + eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x
        else:

            v = sp.calculate_BSpline_1D(x).cuda()
            v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()

            loss = dfdx(pinn, x, t, order=1).cuda() * v \
                + eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x
        #print all components of loss_weak
        logger.debug("Loss weak components:")

        logger.debug(f"dfdx(pinn, x, t, order=1).cuda(): {dfdx(pinn, x, t, order=1).cuda()}")
        # logger.debug(f"v: {v}")
        # logger.debug(f"dfdx(pinn, x, t, order=1).cuda() * v: {dfdx(pinn, x, t, order=1).cuda() * v}")
        # logger.debug(f"eps_interior*dfdx(pinn, x, t, order=1): {eps_interior*dfdx(pinn, x, t, order=1)}")
        # logger.debug(f"v_deriv_x: {v_deriv_x}")
        # logger.debug(f"eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x: {eps_interior*dfdx(pinn, x, t, order=1) * v_deriv_x}")
        # logger.debug(f"Loss weak: {loss}")

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

def interior_loss_colocation(
        pinn: PINN, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        dims: int = 2
        ):

    if dims == 1:
        x = x.cuda()
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

        
def interior_loss_strong(
        pinn: PINN, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        dims: int = 1,
        test_function: B_Splines = None
        ):

    if dims == 1:
        x = x.cuda()

        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        if general_parameters.optimize_test_function:
            v = test_function.calculate_BSpline_1D(x, mode='Adam')
            loss = (
                - eps_interior*dfdx(pinn, x, order=2)
                + dfdx(pinn, x, order=1) 
                ) * v
        else:
            loss = (
                - eps_interior*dfdx(pinn, x, order=2)
                + dfdx(pinn, x, order=1) 
                ) * v
        
       

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


def interior_loss_weak_and_strong(
        pinn: PINN, 
        x:torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None,
        dims: int = 2,
        test_function: B_Splines = None
        ):

    if dims == 1:
        x = x.cuda()
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        if general_parameters.optimize_test_function:
            
            v = test_function.calculate_BSpline_1D(x, mode='Adam').cuda()
            v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode='Adam').cuda()

            loss_weak = (
                dfdx(pinn, x, order=1).cuda() * v
                + eps_interior*dfdx(pinn, x, order=1) * v_deriv_x
                )

            loss_strong = (
                - eps_interior*dfdx(pinn, x, order=1)
                + dfdx(pinn, x, order=2) 
                ) * v
        else:
            v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()

            loss_weak = (
                dfdx(pinn, x, order=1).cuda() * v
                + eps_interior*dfdx(pinn, x, order=1) * v_deriv_x
                )

            loss_strong = (
                - eps_interior*dfdx(pinn, x, order=1)
                + dfdx(pinn, x, order=2) 
                ) * v
        

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


def interior_loss_weak_spline(
        spline: B_Splines, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        mode: str = 'Adam',
        dims: int = 1,
        test_function: B_Splines = None
        ):
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)
    
    if dims == 1:
        x = x.cuda()
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        if general_parameters.optimize_test_function:
            v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode='Adam').cuda()
            v = test_function.calculate_BSpline_1D(x, mode='Adam').cuda()

            loss = spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v \
                + eps_interior*spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v_deriv_x
        else:

            v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x, mode='NN').cuda()

            loss = spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v \
                + eps_interior*spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v_deriv_x

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

def interior_loss_colocation_spline(
        spline: B_Splines, 
        x:torch.Tensor, t: torch.Tensor, 
        sp: B_Splines = None, 
        mode: str = 'Adam',
        dims: int = 1):

    if dims == 1:
        x = x.cuda()
        eps_interior, sp, _, _, v = precalculations_1D(x, sp, colocation = True)

        loss = (spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() - 
                eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()) * v.cuda()

    elif dims == 2:
        eps_interior, sp, _, _, v = precalculations_2D(x, t, sp, colocation=True)

        loss = (spline.calculate_BSpline_2D_deriv_dt(x,t, mode=mode).cpu() - 
                eps_interior*spline.calculate_BSpline_2D_deriv_dtdt(x,t, mode=mode).cpu()
                -eps_interior*spline.calculate_BSpline_2D_deriv_dxdx(x,t, mode=mode).cpu()) * v.cpu()

    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

    return loss.pow(2).mean()

        
def interior_loss_strong_spline(
        spline: B_Splines, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        mode: str = 'Adam',
        dims: int = 1,
        test_function: B_Splines = None):

    

    if dims == 1:

        x = x.cuda()
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        if general_parameters.optimize_test_function:

            v = test_function.calculate_BSpline_1D(x, mode='Adam').cuda()

            loss = (
                - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
                + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
                ) * v
        else:
        
            loss = (
                - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
                + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
                ) * v.cuda()

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


def interior_loss_weak_and_strong_spline(
        spline: B_Splines, 
        x:torch.Tensor, 
        t: torch.Tensor, 
        sp: B_Splines = None, 
        mode: str = 'Adam',
        dims: int = 1,
        test_function: B_Splines = None
        ):

    if dims == 1:
        x = x.cuda()
        eps_interior, sp, _, _, v = precalculations_1D(x, sp)

        if general_parameters.optimize_test_function:
            v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode='Adam').cuda()
            v = test_function.calculate_BSpline_1D(x, mode='Adam').cuda()

            loss_weak = (
                spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v
                + eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda() * v_deriv_x
                )

            loss_strong = (
                - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
                + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
                ) * v.cuda()
            
            print(test_function.coefs)
        else:
        
            v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x, mode='NN').cuda()

            loss_weak = (
                spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda() * v
                + eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda() * v_deriv_x
                )

            loss_strong = (
                - eps_interior*spline.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
                + spline.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
                ) * v.cuda()
        
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


def loss_PINN_learns_coefs(
        pinn: PINN,
        spline: B_Splines, 
        x:torch.Tensor, 
        t: torch.Tensor = None, 
        dims: int = 1,
        ):

    if dims == 1:
        x = x.cuda()
        eps_interior = general_parameters.eps_interior

        # splines have to return matrix function vector for all inputs, so we need to return matrix with dimension
        # input_dim x number_of_coefs == input_dim x number_of_basis_functions
        sp_value = spline._get_basis_functions_1D(x, order=0)
        d_sp_dx = spline._get_basis_functions_1D(x, order=1)
        d2_sp_dx2 = spline._get_basis_functions_1D(x, order=2)

        # pinns return matrix of splines coefficients for all inputs with dimension number_of_coefs x 1
        pinn_value = f(pinn, x)
        d_pinn_dx = dfdx(pinn, x, order=1)
        d2_pinn_dx2 = dfdx(pinn, x, order=2)

        # pinns returns matrix of splines coefficients for all inputs with dimension 
        d_solution_dx = d_pinn_dx @ sp_value + pinn_value @ d_sp_dx
        d2_solution_dx2 = d2_pinn_dx2 @ sp_value + 2*d_pinn_dx @ d_sp_dx + pinn_value @ d2_sp_dx2
        
        loss = d_solution_dx - eps_interior*d2_solution_dx2

    elif dims == 2:
        raise NotImplementedError("So sorry... not implemented yet :c")

    return loss.pow(2).mean()


def boundary_loss_spline(
        spline: B_Splines, 
        x:torch.Tensor, 
        t: torch.Tensor = None,
        dims: int = 1,
        mode: str = 'Adam'):

    if dims == 1:
        boundary_xi = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xi = f_spline(spline, boundary_xi)
        
        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xf = -general_parameters.eps_interior * dfdx_spline(spline, boundary_xf) + f_spline(spline, boundary_xf)-1.0

        return boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()
    
    elif dims == 2:
        t_raw = torch.unique(t).reshape(-1, 1).detach()
        t_raw.requires_grad = True
        
        boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]

        boundary_loss_left = f_spline(spline, boundary_left, t_raw, mode=mode)

        boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]
        
        boundary_loss_right = f_spline(spline, boundary_right, t_raw, mode=mode)

        x_raw = torch.unique(x).reshape(-1, 1).detach()
        x_raw.requires_grad = True

        boundary_top = torch.ones_like(x_raw, requires_grad=True) * t[-1]
        boundary_loss_top = f_spline(spline, boundary_top, x_raw, mode=mode)

        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean() + boundary_loss_top.pow(2).mean()
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

def boundary_loss_PINN_learns_coefs(
        pinn: PINN,
        spline: B_Splines,
        x: torch.Tensor,
        t: torch.Tensor = None,
        dims: int = 2
):

    eps_interior = general_parameters.eps_interior

    if dims == 1:
        

        boundary_xi = x[-1].reshape(-1, 1) #last point = 1
        sp_value_xi = spline._get_basis_functions_1D(boundary_xi, order=0)
        f_value_xi = f(pinn, boundary_xi)
        boundary_loss_xi = f_value_xi @ sp_value_xi
        

        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        sp_value_xf = spline._get_basis_functions_1D(boundary_xf, order=0)
        f_value_xf = f(pinn, boundary_xf)
        f_deriv_value_xf = f(pinn, boundary_xf, order=2)
        sp_deriv_value_xf = spline._get_basis_functions_1D(boundary_xf, order=1)
        boundary_loss_xf = -eps_interior * (f_deriv_value_xf @ sp_value_xf + f_value_xf @ sp_deriv_value_xf) \
                            + f_value_xf @ sp_value_xf - 1.0


        return boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()
    else:
        raise NotImplementedError("Not implemented yet :ccccc")

def boundary_loss(pinn: PINN, 
                  x: torch.Tensor, 
                  t: torch.Tensor = None, 
                  dims: int = 2):

    if dims == 1:
        
        boundary_xi = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xi = f(pinn, boundary_xi)
        
        boundary_xf = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xf = -general_parameters.eps_interior * dfdx(pinn, boundary_xf) + f(pinn, boundary_xf)-1.0

        return boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()
    
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


def initial_loss_spline(spline: B_Splines, x:torch.Tensor, t: torch.Tensor = None):
    # initial condition loss on both the function and its
    # time first-order derivative
    
    x_raw = torch.unique(x).reshape(-1, 1).detach()
    x_raw.requires_grad = True

    f_initial = initial_condition(x_raw)
    t_initial = torch.zeros_like(x_raw)
    t_initial.requires_grad = True

    initial_loss_f = f_spline(spline, x_raw, t_initial) - f_initial 
    initial_loss_df = dfdt_spline(spline, x_raw, t_initial)

    return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()


def initial_loss(pinn: PINN, x:torch.Tensor, t: torch.Tensor = None):
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
    verbose = False, interior_loss_function: Callable = interior_loss_weak_and_strong,
    dims: int = 2,
    test_function=None
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
        if test_function is None:
            final_loss = \
                weight_f * interior_loss_function(pinn, x, t, dims=dims)
        else:
            final_loss = \
                weight_f * interior_loss_function(pinn, x, t, dims=dims, test_function=test_function)
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
    spline: B_Splines, 
    x: torch.Tensor = None, 
    t: torch.Tensor = None, 
    weight_f = 1.0, weight_b = 1.0, weight_i = 1.0, 
    verbose = False, 
    interior_loss_function: Callable = interior_loss_weak_and_strong_spline, 
    dims: int = 1,
    test_function: B_Splines = None
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
        if test_function is None:
            final_loss = \
                weight_f * interior_loss_function(spline, x, t) + \
                weight_b * boundary_loss_spline(spline, x, t)
        else:
            final_loss = \
                weight_f * interior_loss_function(spline, x, t, test_function=test_function) + \
                weight_b * boundary_loss_spline(spline, x, t)
        return final_loss if not verbose else (final_loss, interior_loss_function(spline, x, t), initial_loss_spline(spline, x, t), boundary_loss_spline(spline, x, t))

    elif dims == 2:
        final_loss = \
            weight_f * interior_loss_function(spline, x, t) + \
            weight_i * initial_loss_spline(spline, x, t)

        return final_loss if not verbose else (final_loss, interior_loss_function(spline, x, t), initial_loss_spline(spline, x, t), boundary_loss_spline(spline, x, t))
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")

# TODO
def compute_loss_pinn_learns_coefs():
    pass