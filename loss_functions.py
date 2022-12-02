def interior_loss(pinn: PINN, x:torch.Tensor, t: torch.tensor, sp: B_Splines = splines):
    global eps_interior
    #t here is x in Eriksson problem, x here is y in Erikkson problem
    # loss = dfdt(pinn, x, t, order=1) - eps*dfdt(pinn, x, t, order=2)-eps*dfdx(pinn, x, t, order=2)


    # degree_1, degree_2 = np.random.randint(low=0, high=3, size=2)
    degree_1, degree_2 = 2, 2
    # coef = np.random.randint(low=0, high=2, size=len(sp.knot_vector))
    # coef_2 = np.random.randint(low=0, high=2, size=len(sp.knot_vector))
    coef_float = np.random.rand(len(sp.knot_vector))
    coef_float_2 = np.random.rand(len(sp.knot_vector))
    v = sp.calculate_BSpline_2D(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2)
    v_deriv_x = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    v_deriv_t = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
    loss = torch.trapezoid(torch.trapezoid(
        
        dfdx(pinn, x, t, order=1) * v
        + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
        + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
        
        , dx = 0.01), dx = 0.01)


    return loss.pow(2).mean()

def boundary_loss(pinn: PINN, x:torch.Tensor, t: torch.tensor):
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

def initial_loss(pinn: PINN, x:torch.Tensor, t: torch.tensor):
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
    verbose = False,
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    final_loss = \
        weight_f * interior_loss(pinn, x, t) + \
        weight_i * initial_loss(pinn, x, t)
    
    if not pinn.pinning:
        final_loss += weight_b * boundary_loss(pinn, x, t)

    if not verbose:
        return final_loss
    else:
        return final_loss, interior_loss(pinn, x, t), initial_loss(pinn, x, t), boundary_loss(pinn, x, t)
