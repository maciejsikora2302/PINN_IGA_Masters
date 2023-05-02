from PINN import PINN
import torch
from differential_tools import dfdx, dfdt, f
import numpy as np
from B_Splines import B_Splines
from general_parameters import general_parameters
from typing import Callable, List
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

def initial_condition(x: torch.Tensor) -> torch.Tensor:
    res = torch.sin(torch.pi*x).reshape(-1,1)
    return res


def precalculations(x: torch.Tensor, t: torch.Tensor, generate_test_functions: bool , dims: int = 1):
    degree = general_parameters.spline_degree

    linspace = general_parameters.knot_vector

    # if general_parameters.pinn_is_solution:
    #     x = torch.rand_like(x)
    #     #sort x
    #     x = torch.sort(x, dim=0)[0]
    #     #set first element to 0 and last to 1
    #     x[0] = 0
    #     x[-1] = 1
    #     x = x.unique()
    #     x.requires_grad_(True)

    #     if dims == 2:
    #         t = torch.rand_like(t)

    #         # sort t
    #         t = torch.sort(t, dim=0)[0]

    #         t[0] = 0
    #         t[-1] = 1
    #         t = t.unique()
    #         t.requires_grad_(True)

    
    test_function = B_Splines(linspace, degree, dims=dims) if generate_test_functions else None

    return test_function, x

def _get_loss_basic(**kwargs):
    eps_interior = kwargs["eps_interior"]
    dfdxdx_model = kwargs["dfdxdx_model"]
    dims = kwargs["dims"]
    
    if dims == 1:
        dfdx_model = kwargs["dfdx_model"]

        basic = (
                - eps_interior * dfdxdx_model
                + dfdx_model
                )
    elif dims == 2:
        dfdtdt_model = kwargs["dfdtdt_model"]
        dfdt_model = kwargs["dfdt_model"]

        basic = (
            -eps_interior * (dfdxdx_model + dfdtdt_model)
            + dfdt_model
        )
        
    return basic


def interior_loss_basic(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1, 
        ):
    
    assert dims in [1, 2]
    assert isinstance(model, (PINN, B_Splines))
    # assert isinstance(test_function, B_Splines)
    assert x is not None


    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    _, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = False, \
                                                dims = dims)

    if dims == 1:

        dfdxdx_model = dfdx(model, x, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).to(device)
        dfdx_model = dfdx(model, x, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device)

        basic = _get_loss_basic(
            eps_interior = eps_interior,
            dfdxdx_model = dfdxdx_model,
            dfdx_model = dfdx_model,
            dims=dims
        )
        
        loss = basic.pow(2).mean()
        
        

    elif dims == 2:

        n_x = x.shape[0]
        n_t = t.shape[0]

        dfdxdx_model = dfdx(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dxdx(x, t, mode=mode).to(device)
        dfdtdt_model = dfdt(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dtdt(x, t, mode=mode).to(device)
        dfdt_model = dfdt(model, x, t, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)

        basic = _get_loss_basic(
            eps_interior = eps_interior,
            dfdxdx_model = dfdxdx_model,
            dfdtdt_model = dfdtdt_model,
            dfdt_model = dfdt_model,
            dims = dims
        )

        loss = basic.pow(2)

        loss = torch.trapezoid(torch.trapezoid(loss, dx=1/n_t), dx=1/n_x)

    return loss

def _get_loss_weak(**kwargs):
    eps_interior = kwargs["eps_interior"]
    v = kwargs["v"]
    v_deriv_x = kwargs["v_deriv_x"]
    dims = kwargs["dims"]
    dfdx_model = kwargs["dfdx_model"]

    if dims == 1:
        v_at_first_point = kwargs["v_at_first_point"]
        model_value_at_first_point = kwargs["model_value_at_first_point"]

        weak = (dfdx_model * v \
                + eps_interior * dfdx_model * v_deriv_x).mean() \
                + model_value_at_first_point * v_at_first_point \
                - v_at_first_point
    elif dims == 2:
        dfdt_model = kwargs["dfdt_model"]
        sin_pi_x = kwargs["sin_pi_x"]
        cos_pi_x = kwargs["cos_pi_x"]
        v_deriv_t = kwargs["v_deriv_t"]
        pi = torch.pi

        b_uv = (eps_interior * (dfdx_model * v_deriv_x + dfdt_model * v_deriv_t) \
                + dfdt_model * v)
        
        I_v = (eps_interior * (sin_pi_x * dfdt_model - pi * cos_pi_x * dfdx_model) \
               + sin_pi_x * v)
        
        weak = b_uv - I_v
        
    return weak

def interior_loss_weak(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1, 
        ):
    
    assert dims in [1, 2], "Only 1D and 2D are supported"
    assert isinstance(model, (PINN, B_Splines)), "model must be a PINN or a B_Splines"
    assert x is not None, "x must be a tensor"

    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)
    test_function = test_function if general_parameters.optimize_test_function else generated_test_function

    v =                     test_function.calculate_BSpline_1D(x, mode=mode).to(device)          if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode).to(device)
    v_deriv_x =             test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device) if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).to(device)

    if dims == 1:

        v_at_first_point = test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode)
        model_value_at_first_point = f(model, x[0].reshape(-1, 1))
        dfdx_model = dfdx(model, x, order=1).to(device)

        weak = _get_loss_weak(
            eps_interior = eps_interior,
            v = v,
            v_deriv_x = v_deriv_x,
            v_at_first_point = v_at_first_point,
            dfdx_model = dfdx_model,
            model_value_at_first_point = model_value_at_first_point,
            dims = dims
        )

        # Calculate loss as the square of b_weak
        loss = weak.pow(2)

        loss = loss.mean()

    elif dims == 2:
        # Calculate the number of points in x and t
        n_x = x.shape[0]
        n_t = t.shape[0]
        
        v_deriv_t = test_function.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)
        dfdt_model = dfdt(model, x, t, order=1).to(device)
        dfdx_model = dfdx(model, x, t, order=1).to(device)
        sin_pi_x = initial_condition(x).to(device)
        cos_pi_x = torch.cos(torch.pi * x).to(device)

        weak = _get_loss_weak(
            eps_interior = eps_interior,
            v = v,
            v_deriv_x = v_deriv_x,
            v_deriv_t = v_deriv_t,
            dfdx_model = dfdx_model,
            dfdt_model = dfdt_model,
            sin_pi_x = sin_pi_x,
            cos_pi_x = cos_pi_x,
            dims = dims
        )

        
        loss = torch.trapezoid(
            torch.trapezoid(
                        weak,
                dx=1/n_x),
            dx=1/n_t)
        
        loss = loss.pow(2)

    return loss


def _get_loss_strong(**kwargs):
    eps_interior = kwargs["eps_interior"]
    v = kwargs["v"]
    dfdxdx_model = kwargs["dfdxdx_model"]
    dims = kwargs["dims"]

    if dims == 1:
        dfdx_model = kwargs["dfdx_model"]

        strong = (
                - eps_interior * dfdxdx_model
                + dfdx_model
                ) * v
    elif dims == 2:

        dfdt_model = kwargs["dfdt_model"]
        dfdtdt_model = kwargs["dfdtdt_model"]

        strong = (
                - eps_interior * (dfdxdx_model + dfdtdt_model)
                + dfdt_model
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
    # assert isinstance(test_function, B_Splines)
    assert x is not None


    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)
    test_function = test_function if general_parameters.optimize_test_function else generated_test_function

    if dims == 1:

        v = test_function.calculate_BSpline_1D(x, mode=mode).to(device)
        dfdxdx_model = dfdx(model, x, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).to(device)
        dfdx_model = dfdx(model, x, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device)

        strong = _get_loss_strong(
            eps_interior = eps_interior,
            dfdxdx_model = dfdxdx_model,
            dfdx_model = dfdx_model,
            v = v,
            dims=dims
        )
        
        loss = strong.pow(2).mean()
        
        

    elif dims == 2:

        n_x = x.shape[0]
        n_t = t.shape[0]

        v = test_function.calculate_BSpline_2D(x, t, mode=mode).to(device)
        dfdtdt_model = dfdt(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dtdt(x, t, mode=mode).to(device)
        dfdxdx_model = dfdx(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dxdx(x, t, mode=mode).to(device)
        dfdt_model = dfdt(model, x, t, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)

        strong = _get_loss_strong(
            eps_interior = eps_interior,
            dfdxdx_model = dfdxdx_model,
            dfdtdt_model = dfdtdt_model,
            dfdt_model = dfdt_model,
            v = v,
            dims = dims
        )

        loss = torch.trapezoid(torch.trapezoid(strong.pow(2), dx=1/n_x), dx=1/n_t)

    return loss




def interior_loss_weak_and_strong(
        model,
        x: torch.Tensor, 
        t: torch.Tensor, 
        test_function: B_Splines = None,
        dims: int = 1
    ):

    assert dims in [1, 2]
    assert isinstance(model, (PINN, B_Splines, List[PINN]))
    assert isinstance(test_function, B_Splines) or test_function is None
    assert x is not None

    mode = "Adam" if general_parameters.optimize_test_function else "NN"

    eps_interior = general_parameters.eps_interior
    generated_test_function, x = precalculations(\
                                                x = x, t = t, \
                                                generate_test_functions = True if not general_parameters.optimize_test_function else False, \
                                                dims = dims)

    test_function = test_function if general_parameters.optimize_test_function else generated_test_function


    if not general_parameters.pinn_learns_coeff:

        v =         test_function.calculate_BSpline_1D(x, mode=mode).to(device)          if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode).to(device)
        v_deriv_x = test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device) if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).to(device)

        if dims == 1:

            dfdx_model = dfdx(model, x, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device)
            dfdxdx_model = dfdx(model, x, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).to(device)
            model_value_at_first_point = f(model, x[0].reshape(-1, 1))
            v_at_first_point = test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode)


            strong = _get_loss_strong(
                eps_interior = eps_interior,
                dfdxdx_model = dfdxdx_model,
                dfdx_model = dfdx_model,
                v = v,
                dims=dims
            )

            weak = _get_loss_weak(
                eps_interior = eps_interior,
                v = v,
                v_deriv_x = v_deriv_x,
                v_at_first_point = v_at_first_point,
                dfdx_model = dfdx_model,
                model_value_at_first_point = model_value_at_first_point,
                dims = dims
            )

            loss = weak.pow(2) + strong.pow(2)
            loss = loss.mean()


        elif dims == 2:
            # Calculate the number of points in x and t
            n_x = x.shape[0]
            n_t = t.shape[0]

            dfdtdt_model = dfdt(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dtdt(x, t, mode=mode).to(device)
            dfdxdx_model = dfdx(model, x, t, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dxdx(x, t, mode=mode).to(device)
            dfdt_model = dfdt(model, x, t, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)
            v_deriv_t = test_function.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)
            dfdt_model = dfdt(model, x, t, order=1).to(device)
            dfdx_model = dfdx(model, x, t, order=1).to(device)
            sin_pi_x = initial_condition(x).to(device)
            cos_pi_x = torch.cos(torch.pi * x).to(device)

            strong = _get_loss_strong(
                eps_interior = eps_interior,
                dfdxdx_model = dfdxdx_model,
                dfdtdt_model = dfdtdt_model,
                dfdt_model = dfdt_model,
                v = v,
                dims = dims
            )

            weak = _get_loss_weak(
                eps_interior = eps_interior,
                v = v,
                v_deriv_x = v_deriv_x,
                v_deriv_t = v_deriv_t,
                dfdx_model = dfdx_model,
                dfdt_model = dfdt_model,
                sin_pi_x = sin_pi_x,
                cos_pi_x = cos_pi_x,
                dims = dims
            )

            loss_weak = torch.trapezoid(
                torch.trapezoid(
                    weak,
                    dx=1/n_t
                ),
                dx=1/n_x
            )
            loss_weak = loss_weak.pow(2)
            
            strong = strong.pow(2)

            loss_strong = torch.trapezoid(
                torch.trapezoid(
                    strong,
                    dx = 1/n_x
                ),
                dx = 1/n_t
            )

            loss = loss_weak + loss_strong

        return loss

    elif general_parameters.pinn_learns_coeff:
        #here model is a list of pinns
        models = model

        if dims == 1:

            # splines have to return matrix function vector for all inputs, so we need to return matrix with dimension
            # input_dim x number_of_coeffs
            sp_value = test_function._get_basis_functions_1D(x, order=0)
            d_sp_dx = test_function._get_basis_functions_1D(x, order=1)
            d2_sp_dx2 = test_function._get_basis_functions_1D(x, order=2)

            n_eps = len(general_parameters.epsilon_list)
            n_coeffs = general_parameters.n_coeff

            # Initialize matrix of zeros for storing pinns' values for different values of epsilon
            pinn_value = torch.zeros(
                n_eps,
                n_coeffs
            )
            # pinns will form matrix of splines coefficients with dimension number_of_epsilons x number_of_coeffs
            pinn_value = f(models[0], general_parameters.epsilon_list)
            for pinn in models[1:]:
                pinn_value = torch.cat((
                    pinn_value,
                    f(pinn, general_parameters.epsilon_list).flatten().unsqueeze(1)
                ), dim=1)

            solution = pinn_value @  sp_value
            d_solution_dx = pinn_value @ d_sp_dx 
            d2_solution_dx2 = pinn_value @ d2_sp_dx2
            
            #TODO: change this to use _get_loss_weak and _get_loss_strong
            loss_weak = (d_solution_dx * v \
                + general_parameters.epsilon_list * d_solution_dx * v_deriv_x).mean() \
                + solution * v_at_first_point \
                - v_at_first_point
            
            loss_strong = (
                - general_parameters.epsilon_list * d2_solution_dx2
                + d_solution_dx 
                ) * v
            
            _get_loss_strong(
                eps_interior = eps_interior,
                dfdxdx_model = dfdxdx_model,
                dfdx_model = dfdx_model,
                v = v
            )

        elif dims == 2:
            raise NotImplementedError("So sorry... not implemented yet :c")

        return (loss_weak.pow(2) + loss_strong.pow(2)).mean()


def boundary_loss(
        model, 
        x: torch.Tensor, 
        t: torch.Tensor = None, 
        dims: int = 1,
        spline:B_Splines = None):

    assert dims in [1, 2], "dims must be 1 or 2"
    assert isinstance(model, (PINN, B_Splines, List[PINN])), "model must be PINN or B_Splines"

    mode = "NN" if isinstance(model, PINN) else "Adam"

    if not general_parameters.pinn_learns_coeff:
        if dims == 1:
            
            boundary_xi = x[-1].reshape(-1, 1) #last point = 1
            boundary_loss_xi = f(model, boundary_xi) 
            boundary_xf = x[0].reshape(-1, 1) #first point = 0

            dfdx_model = dfdx(model, boundary_xf, order=1)
            f_model = f(model, boundary_xf)
            
            boundary_loss_xf = -general_parameters.eps_interior * dfdx_model + f_model-1.0

            return boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()
        
        else:
            t_raw = torch.unique(t).reshape(-1, 1).detach()
            t_raw.requires_grad = True

            x_raw = torch.unique(x).reshape(-1, 1).detach()
            x_raw.requires_grad = True
            
            # LOSS(0, t)
            boundary_left = torch.ones_like(x_raw, requires_grad=True) * x[0] #zeros
            boundary_loss_left = f(model, boundary_left, t_raw)

            # LOSS(x, 1)
            boundary_right = torch.ones_like(t_raw, requires_grad=True) * t[-1] #ones
            boundary_loss_right = f(model, x_raw, boundary_right)

            # LOSS(1, t)
            boundary_top = torch.ones_like(x_raw, requires_grad=True) * x[-1]
            boundary_loss_top = f(model, boundary_top, t_raw)

            return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean() + boundary_loss_top.pow(2).mean()
    else:
        models = model
        if dims == 1:
            boundary_xi = x[0].reshape(-1, 1) # first point = 0
            d_sp_value_xi = spline._get_basis_functions_1D(boundary_xi, order=1).flatten()[0] # We take derivative of first spline in x=1
            f_value_xi = f(models[0], general_parameters.epsilon_list)
            boundary_loss_xi = -general_parameters.epsilon_list * f_value_xi * d_sp_value_xi \
                                + f_value_xi - 1.0
            
            boundary_loss_xf = f(models[-1], general_parameters.epsilon_list)

            return boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()
        else:
            raise NotImplementedError("Not implemented yet :ccccc")        

def initial_loss(model, x:torch.Tensor, t: torch.Tensor = None):

    assert isinstance(model, (PINN, B_Splines)), "model must be PINN or B_Splines"
    
    x_raw = torch.unique(x).reshape(-1, 1).detach()
    x_raw.requires_grad = True

    # t_initial = torch.zeros_like(x_raw).to(device)
    # t_initial.requires_grad = True

    t_initial = torch.Tensor([0]).to(device)
    t_initial.requires_grad = True

    f_initial = initial_condition(x_raw)

    initial_loss_f = f(model, x_raw, t_initial) - f_initial 
    # initial_loss_df = dfdt(model, x_raw, t_initial, order=1)

    # LOSS(x, 0)
    return initial_loss_f.pow(2).mean()

def compute_loss(
    model,
    x: torch.Tensor = None, 
    t: torch.Tensor = None, 
    weight_f = 1.0, 
    weight_b = 1.0, 
    weight_i = 1.0, 
    interior_loss_function: Callable = interior_loss_weak_and_strong,
    dims: int = 2,
    test_function=None
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    assert dims in [1, 2], "dims must be 1 or 2"
    assert isinstance(model, (PINN, B_Splines)), "model must be PINN or B_Splines"

    final_loss = \
        weight_f * interior_loss_function(model, x, t, dims=dims, test_function=test_function)

    if dims == 2:
        final_loss += weight_i * initial_loss(model, x, t)
        
    final_loss += weight_b * boundary_loss(model, x, t, dims=dims)

    return final_loss