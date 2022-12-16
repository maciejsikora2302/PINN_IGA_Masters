import torch
from loss_functions import interior_loss_colocation, precalculations
from B_Splines import B_Splines
from PINN import PINN
import general_parameters

def interrior_loss_colocation_deriv_coefs(pinn: PINN, x: torch.Tensor, t: torch.Tensor): # The gradient is calculated w.r. to wiegths of a test function
    
    _, sp, degree_1, degree_2, _, _, _ = precalculations(x, t, sp)
    coef_float = general_parameters.test_function_weight_x
    coef_float_2 = general_parameters.test_function_weight_t

    loss_deriv_coef = sp.calculate_BSpline_1D(t.detach(), coef_float_2, degree_2).sum() * sp.calculate_BSpline_1D(x.detach(), coef_float, degree_1)
    loss_deriv_coef_2 = sp.calculate_BSpline_1D(x.detach(), coef_float, degree_1).sum() * sp.calculate_BSpline_1D(t.detach(), coef_float_2, degree_2)
    loss_colocation = interior_loss_colocation(pinn, x, t, sp, extended=True)

    # Gradient with respect to the B(x) weights
    loss_deriv_coef_final = 2 * loss_colocation * loss_deriv_coef

    # Gradient with respect to B(t) weights
    loss_deriv_coef_2_final = 2 * loss_colocation * loss_deriv_coef_2

    return loss_deriv_coef_final.pow(2).mean(), loss_deriv_coef_2_final.pow(2).mean()
