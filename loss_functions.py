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

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(math.pi*x).reshape(-1,1)
    return res


def precalculations(x: torch.Tensor, t: torch.Tensor, generate_test_functions: bool , dims: int = 1):
    degree = general_parameters.spline_degree
    coefs_vector_length = general_parameters.n_coeff

    linspace = general_parameters.knot_vector

    if general_parameters.pinn_is_solution:
        x = torch.rand_like(x)
        #sort x
        x = torch.sort(x, dim=0)[0]
        #set first element to 0 and last to 1
        x[0] = 0
        x[-1] = 1
        x.requires_grad_(True)

    #coefs random floats between 0 and 1 as a tensor
    coefs = torch.Tensor(np.random.rand(coefs_vector_length))

    
    test_function = B_Splines(linspace, degree, coefs=coefs, dims=dims) if generate_test_functions else None

    return test_function, x

def _get_loss_basic(**kwargs):
    eps_interior = kwargs["eps_interior"]
    dfdx_model = kwargs["dfdx_model"]
    dfdxdx_model = kwargs["dfdxdx_model"]
    basic = (
            - eps_interior * dfdxdx_model
            + dfdx_model
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
            dfdx_model = dfdx_model
        )
        
        loss = basic.pow(2).mean()
        
        

    elif dims == 2:

        n_x = x.shape[0]
        n_t = t.shape[0]


        raise Exception("Implement 2D interior loss basic")

        loss = torch.trapezoid(torch.trapezoid(

            ((dfdt(pinn, x, t, order=1).to(device_cpu) 
                            - eps_interior*dfdt(pinn, x, t, order=2).to(device_cpu)
                            - eps_interior*dfdx(pinn, x, t, order=2).to(device_cpu)) * v.to(device_cpu)).pow(2)

                            , dx=1/n_x), dx=1/n_t)

    return loss

def _get_loss_weak(**kwargs):
    eps_interior = kwargs["eps_interior"]
    v = kwargs["v"]
    v_deriv_x = kwargs["v_deriv_x"]
    v_at_first_point = kwargs["v_at_first_point"]
    dfdx_model = kwargs["dfdx_model"]
    model_value_at_first_point = kwargs["model_value_at_first_point"]


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

    v =                     test_function.calculate_BSpline_1D(x, mode=mode)                    if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode)
    v_deriv_x =             test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode)           if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode)
    v_at_first_point =      test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode)  if dims == 1 else test_function.calculate_BSpline_2D(x[0].reshape(-1, 1), t[0].reshape(-1, 1), mode=mode)

    if dims == 1:
        
        dfdx_model = dfdx(model, x, order=1).to(device)
        model_value_at_first_point = f(model, x[0].reshape(-1, 1))

        weak = _get_loss_weak(
            eps_interior = eps_interior,
            v = v,
            v_deriv_x = v_deriv_x,
            v_at_first_point = v_at_first_point,
            dfdx_model = dfdx_model,
            model_value_at_first_point = model_value_at_first_point
        )

        # Calculate loss as the square of b_weak
        loss = weak.pow(2)

    elif dims == 2:
        # Calculate the number of points in x and t
        n_x = x.shape[0]
        n_t = t.shape[0]
        
        v_deriv_t = test_function.calculate_BSpline_2D_deriv_dt(x, t, mode=mode).to(device)

        dfdt_model = dfdt(model, x, t, order=1)
        dfdx_model = dfdx(model, x, t, order=1)
        # Calculate the loss using the variables above
        loss = torch.trapezoid(
            torch.trapezoid(
                (dfdt_model * v
                + eps_interior*dfdx_model * v_deriv_x
                + eps_interior*dfdt_model * v_deriv_t).pow(2),
                dx=1/n_x),
            dx=1/n_t)

    return loss


def _get_loss_strong(**kwargs):
    eps_interior = kwargs["eps_interior"]
    v = kwargs["v"]
    dfdx_model = kwargs["dfdx_model"]
    dfdxdx_model = kwargs["dfdxdx_model"]
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
            v = v
        )
        
        loss = strong.pow(2).mean()
        
        

    elif dims == 2:

        n_x = x.shape[0]
        n_t = t.shape[0]


        raise Exception("Implement 2D interior loss strong")

        loss = torch.trapezoid(torch.trapezoid(

            ((dfdt(pinn, x, t, order=1).to(device_cpu) 
                            - eps_interior*dfdt(pinn, x, t, order=2).to(device_cpu)
                            - eps_interior*dfdx(pinn, x, t, order=2).to(device_cpu)) * v.to(device_cpu)).pow(2)

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

    v =                     test_function.calculate_BSpline_1D(x, mode=mode).to(device)                     if dims == 1 else test_function.calculate_BSpline_2D(x, t, mode=mode).to(device)
    v_deriv_x =             test_function.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device)            if dims == 1 else test_function.calculate_BSpline_2D_deriv_dx(x, t, mode=mode).to(device)
    v_at_first_point =      test_function.calculate_BSpline_1D(x[0].reshape(-1, 1), mode=mode).to(device)   if dims == 1 else test_function.calculate_BSpline_2D(x[0].reshape(-1, 1), t[0].reshape(-1, 1), mode=mode).to(device)


    if not general_parameters.pinn_learns_coeff:
        if dims == 1:
            
            dfdx_model = dfdx(model, x, order=1).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dx(x, mode=mode).to(device)
            dfdxdx_model = dfdx(model, x, order=2).to(device) if isinstance(model, PINN) else model.calculate_BSpline_1D_deriv_dxdx(x, mode=mode).to(device)
            model_value_at_first_point = f(model, x[0].reshape(-1, 1))

            weak = _get_loss_weak(
                eps_interior = eps_interior,
                v = v,
                v_deriv_x = v_deriv_x,
                v_at_first_point = v_at_first_point,
                dfdx_model = dfdx_model,
                model_value_at_first_point = model_value_at_first_point
            )
            
            strong = _get_loss_strong(
                eps_interior = eps_interior,
                dfdxdx_model = dfdxdx_model,
                dfdx_model = dfdx_model,
                v = v
            )

            del dfdx_model, dfdxdx_model



        elif dims == 2:
            # Calculate the number of points in x and t
            n_x = x.shape[0]
            n_t = t.shape[0]

            raise Exception("Implement 2D interior loss weak and strong")
            
            dfdt_model = dfdt(model, x, t, order=1).to(device_cpu)
            dfdx_model = dfdx(model, x, t, order=1).to(device_cpu)
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
            
            boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
            boundary_loss_left = f(model, boundary_left, t_raw)

            boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]

            boundary_loss_right = f(model, boundary_right, t_raw)

            x_raw = torch.unique(x).reshape(-1, 1).detach()
            x_raw.requires_grad = True

            boundary_top = torch.ones_like(x_raw, requires_grad=True) * t[-1]
            boundary_loss_top = f(model, boundary_top, x_raw)

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

    # initial condition loss on both the function and its
    # time first-order derivative
    x_raw = torch.unique(x).reshape(-1, 1).detach()
    x_raw.requires_grad = True

    f_initial = initial_condition(x_raw)
    t_initial = torch.zeros_like(x_raw)
    t_initial.requires_grad = True

    initial_loss_f = f(model, x_raw, t_initial) - f_initial 
    initial_loss_df = dfdt(model, x_raw, t_initial, order=1)

    return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()

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
        final_loss += weight_i * initial_loss(model, x, t, dims=dims)
        

    final_loss += weight_b * boundary_loss(model, x, t, dims=dims)

    # print(final_loss)

    return final_loss