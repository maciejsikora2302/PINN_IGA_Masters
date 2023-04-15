from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f, f_spline, dfdx_spline, dfdt_spline
import numpy as np
from B_Splines import B_Splines
from general_parameters import general_parameters, logger
from typing import Callable, List
import math

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(math.pi*x).reshape(-1,1)
    return res


def precalculations(x: torch.Tensor, t: torch.Tensor, generate_test_functions: bool , dims: int = 1):
    degree = general_parameters.spline_degree
    coefs_vector_length = general_parameters.n_coeff

    linspace = general_parameters.knot_vector

    if general_parameters.pinn_is_solution:
        x = torch.rand_like(x)
        x.requires_grad_(True)

    #coefs random floats between 0 and 1 as a tensor
    coefs = torch.Tensor(np.random.rand(coefs_vector_length))

    if generate_test_functions:
        test_function = B_Splines(linspace, degree, coefs=coefs, dims=dims) if test_function is None else test_function

    return test_function, x

def _get_loss_weak(eps_interior, v, v_deriv_x, v_at_first_point, dfdx_model, model_value_at_first_point):
    weak = (dfdx_model * v \
            + eps_interior * dfdx_model * v_deriv_x).mean() \
            + model_value_at_first_point * v_at_first_point \
            - v_at_first_point
        
    return weak

def interior_loss_weak(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1, 
        ):
    
    assert dims in [1, 2]
    assert isinstance(model, (PINN, B_Splines))
    assert isinstance(test_function, B_Splines)
    assert x is not None

    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)
    test_function = test_function if general_parameters.optimize_test_function else generated_test_function

    v =                     test_function.calculate_BSpline_1D(x, mode=mode).cuda()                     if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode).cuda()
    v_deriv_x =             test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()            if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).cuda()
    v_at_first_point =      test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode).cuda()   if dims == 1 else test_function.calculate_BSpline_2D(x[0].reshape(-1, 1), t[0].reshape(-1, 1), mode=mode).cuda()

    if dims == 1:
        
        dfdx_model = dfdx(model, x, order=1).cuda() if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
        model_value_at_first_point = f(model, x[0].reshape(-1, 1)) if isinstance(model, PINN) else f_spline(model, x[0].reshape(-1, 1), mode=mode)

        weak = _get_loss_weak(eps_interior, v, v_deriv_x, v_at_first_point, dfdx_model, model_value_at_first_point)

        # Calculate loss as the square of b_weak
        loss = weak.pow(2)

    elif dims == 2:
        # Calculate the number of points in x and t
        n_x = x.shape[0]
        n_t = t.shape[0]
        
        v_deriv_t = test_function.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).cuda()

        dfdt_model = dfdt(model, x, t, order=1).cpu()
        dfdx_model = dfdx(model, x, t, order=1).cpu()
        # Calculate the loss using the variables above
        loss = torch.trapezoid(
            torch.trapezoid(
                (dfdt_model * v
                + eps_interior*dfdx_model * v_deriv_x
                + eps_interior*dfdt_model * v_deriv_t).pow(2),
                dx=1/n_x),
            dx=1/n_t)

    return loss


def _get_loss_strong(eps_interior, dfdxdx_model, dfdx_model, v):
    strong = (
            - eps_interior * dfdxdx_model
            + dfdx_model
            ) * v
        
    return strong

def interior_loss_strong(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1, 
        ):
    
    assert dims in [1, 2]
    assert isinstance(model, (PINN, B_Splines))
    assert isinstance(test_function, B_Splines)
    assert x is not None


    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)
    test_function = test_function if general_parameters.optimize_test_function else generated_test_function

    if dims == 1:

        v = test_function.calculate_BSpline_1D(x, mode=mode).cuda()
        
        dfdxdx_model = dfdx(model, x, order=2).cuda() if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
        dfdx_model = dfdx(model, x, order=1).cuda() if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()

        strong = _get_loss_strong(eps_interior, dfdxdx_model, dfdx_model, v)
        
        loss = strong.pow(2).mean()
        
        

    elif dims == 2:

        n_x = x.shape[0]
        n_t = t.shape[0]


        raise Exception("Implement 2D interior loss strong")

        loss = torch.trapezoid(torch.trapezoid(

            ((dfdt(pinn, x, t, order=1).cpu() 
                            - eps_interior*dfdt(pinn, x, t, order=2).cpu()
                            - eps_interior*dfdx(pinn, x, t, order=2).cpu()) * v.cpu()).pow(2)

                            , dx=1/n_x), dx=1/n_t)

    return loss





def interior_loss_weak_and_strong(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1
    ):

    assert dims in [1, 2]
    assert isinstance(model, (PINN, B_Splines))
    assert isinstance(test_function, B_Splines)
    assert x is not None

    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)
    test_function = test_function if general_parameters.optimize_test_function else generated_test_function

    v =                     test_function.calculate_BSpline_1D(x, mode=mode).cuda()                     if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode).cuda()
    v_deriv_x =             test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()            if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).cuda()
    v_at_first_point =      test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode).cuda()   if dims == 1 else test_function.calculate_BSpline_2D(x[0].reshape(-1, 1), t[0].reshape(-1, 1), mode=mode).cuda()


    if dims == 1:
        
        dfdx_model = dfdx(model, x, order=1).cuda() if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).cuda()
        dfdxdx_model = dfdx(model, x, order=2).cuda() if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).cuda()
        model_value_at_first_point = f(model, x[0].reshape(-1, 1)) if isinstance(model, PINN) else f_spline(model, x[0].reshape(-1, 1), mode=mode)

        weak = _get_loss_weak(eps_interior, v, v_deriv_x, v_at_first_point, dfdx_model, model_value_at_first_point)
        strong = _get_loss_strong(eps_interior, dfdxdx_model, dfdx_model, v)


    elif dims == 2:
        # Calculate the number of points in x and t
        n_x = x.shape[0]
        n_t = t.shape[0]

        raise Exception("Implement 2D interior loss weak and strong")
        
        dfdt_model = dfdt(model, x, t, order=1).cpu()
        dfdx_model = dfdx(model, x, t, order=1).cpu()
        # Calculate the loss using the variables above
        loss = torch.trapezoid(
            torch.trapezoid(
                (dfdt_model * v
                + eps_interior*dfdx_model * v_deriv_x
                + eps_interior*dfdt_model * v_deriv_t).pow(2),
                dx=1/n_x),
            dx=1/n_t)
    
    loss = weak.pow(2) + strong.pow(2)
    loss = loss.mean()

    return loss







def loss_PINN_learns_coeff(
        pinn_list: List[PINN],
        spline: B_Splines, 
        x:torch.Tensor, 
        t: torch.Tensor = None, 
        sp: B_Splines = None,
        dims: int = 1,
        test_function: B_Splines = None
        ):

    
    eps_interior = general_parameters.eps_interior
    
    if dims == 1:
        x = x.cuda()

        _, sp, _, _, v, x = precalculations_1D(x, sp)
        

        # splines have to return matrix function vector for all inputs, so we need to return matrix with dimension
        # input_dim x number_of_coeffs
        sp_value = spline._get_basis_functions_1D(x, order=0)
        d_sp_dx = spline._get_basis_functions_1D(x, order=1)
        d2_sp_dx2 = spline._get_basis_functions_1D(x, order=2)

        n_eps = len(general_parameters.epsilon_list)
        n_coeffs = general_parameters.n_coeff

        # Initialize matrix of zeros for storing pinns' values for different values of epsilon
        pinn_value = torch.zeros(
            n_eps,
            n_coeffs
        )
        # pinns will form matrix of splines coefficients with dimension number_of_epsilons x number_of_coeffs
        pinn_value = f(pinn_list[0], general_parameters.epsilon_list)

        for pinn in pinn_list[1:]:
            temp = f(pinn, general_parameters.epsilon_list) # Dimension == 1 x number_of_epsilons
            pinn_value = torch.cat((
                pinn_value,
                temp.flatten().unsqueeze(1)
            ), dim=1)

        solution = pinn_value @  sp_value
        d_solution_dx = pinn_value @ d_sp_dx 
        d2_solution_dx2 = pinn_value @ d2_sp_dx2
        
        if general_parameters.optimize_test_function:
            
            v = test_function.calculate_BSpline_1D(x, mode='Adam').cuda()
            v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode='Adam').cuda()

            first_point = x[0].reshape(-1, 1) #first point = 0

            # v_at_last_point = test_function.calculate_BSpline_1D(last_point, mode="Adam").cuda()
            v_at_first_point = test_function.calculate_BSpline_1D(first_point, mode="Adam").cuda()

            loss_weak = (d_solution_dx * v \
                + general_parameters.epsilon_list * d_solution_dx * v_deriv_x).mean() \
                + solution * v_at_first_point \
                - v_at_first_point
            
            loss_strong = (
                - general_parameters.epsilon_list * d2_solution_dx2
                + d_solution_dx 
                ) * v
        else:
            v = sp.calculate_BSpline_1D(x).cuda()
            v_deriv_x = sp.calculate_BSpline_1D_deriv_dx(x).cuda()

            first_point = x[0].reshape(-1, 1) #first point = 0

            # v_at_last_point = test_function.calculate_BSpline_1D(last_point, mode="Adam").cuda()
            v_at_first_point = sp.calculate_BSpline_1D(first_point).cuda()

            loss_weak = (d_solution_dx * v \
                + general_parameters.epsilon_list * d_solution_dx * v_deriv_x).mean() \
                + solution * v_at_first_point \
                - v_at_first_point
            
            loss_strong = (
                - general_parameters.epsilon_list*d2_solution_dx2
                + d_solution_dx 
                ) * v
        

    elif dims == 2:
        raise NotImplementedError("So sorry... not implemented yet :c")

    return (loss_weak.pow(2) + loss_strong.pow(2)).mean()


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

def boundary_loss_PINN_learns_coeff(
        pinn_list: List[PINN],
        spline: B_Splines,
        x: torch.Tensor,
        t: torch.Tensor = None,
        dims: int = 2
):

    if dims == 1:

        boundary_xi = x[0].reshape(-1, 1) # first point = 0
        d_sp_value_xi = spline._get_basis_functions_1D(boundary_xi, order=1).flatten()[0] # We take derivative of first spline in x=1
        f_value_xi = f(pinn_list[0], general_parameters.epsilon_list)
        boundary_loss_xi = -general_parameters.epsilon_list * f_value_xi * d_sp_value_xi \
                            + f_value_xi - 1.0
        
        boundary_loss_xf = f(pinn_list[-1], general_parameters.epsilon_list)

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

        return final_loss

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
    
def compute_loss_PINN_learns_coeff(
    pinn_list: List[PINN], spline: B_Splines,
    x: torch.Tensor = None, t: torch.Tensor = None, 
    weight_f = 1.0, weight_b = 1.0,
    dims: int = 2,
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
        final_loss = \
            weight_f * loss_PINN_learns_coeff(pinn_list, spline, x, t, dims=dims, test_function=test_function) \
            + weight_b * boundary_loss_PINN_learns_coeff(pinn_list, spline, x, t, dims=dims)
        
        return final_loss
    
    elif dims == 2:
        raise NotImplementedError("Not implemented yet!")
    else:
        raise ValueError("Wrong dimensionality, must be 1 or 2")