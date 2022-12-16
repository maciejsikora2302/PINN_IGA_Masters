import torch
from loss_functions import interior_loss_colocation, precalculations_2D, precalculations_1D
from differential_tools import dfdt, dfdx
from PINN import PINN
from general_parameters import general_parameters

def interrior_loss_colocation_deriv_coefs(pinn: PINN, x: torch.Tensor, t: torch.Tensor, dims: int = 2): # The gradient is calculated w.r. to wiegths of a test function
    
    eps_interior = general_parameters.eps_interior

    if dims == 1:
        _, sp, degree_1, _, _, _, _ = precalculations_1D(x, t, sp)
        diff_equation = dfdx(pinn, x, order=1) - eps_interior * dfdx(pinn, x, order=2)
        coef_ones = torch.ones(general_parameters.knot_vector_length)

        loss_deriv_coef =  sp.calculate_BSpline_1D(x.detach(), coef_ones, degree_1)
        loss_colocation = interior_loss_colocation(pinn, x, t, sp, extended=True)

        # Gradient with respect to the B(x) weights
        loss_deriv_coef_final = 2 * loss_colocation * loss_deriv_coef * diff_equation

        return loss_deriv_coef_final.pow(2).mean(), None
        
    elif dims == 2:
        _, sp, degree_1, degree_2, _, _, _ = precalculations_2D(x, t, sp)
        coef_float = general_parameters.test_function_weight_x
        coef_float_2 = general_parameters.test_function_weight_t
        coef_ones = torch.ones(general_parameters.knot_vector_length)

        diff_equation = dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)

        loss_deriv_coef = sp.calculate_BSpline_1D(t.detach(), coef_float_2, degree_2).sum() * sp.calculate_BSpline_1D(x.detach(), coef_ones, degree_1)
        loss_deriv_coef_2 = sp.calculate_BSpline_1D(x.detach(), coef_float, degree_1).sum() * sp.calculate_BSpline_1D(t.detach(), coef_ones, degree_2)
        loss_colocation = interior_loss_colocation(pinn, x, t, sp, extended=True)

        # Gradient with respect to the B(x) weights
        loss_deriv_coef_final = 2 * loss_colocation * loss_deriv_coef * diff_equation

        # Gradient with respect to B(t) weights
        loss_deriv_coef_2_final = 2 * loss_colocation * loss_deriv_coef_2 * diff_equation

        return loss_deriv_coef_final.pow(2).mean(), loss_deriv_coef_2_final.pow(2).mean()
    else:
        raise ValueError("The dimension of the problem must be 1 or 2")