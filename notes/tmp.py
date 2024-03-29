# colocation pinn (3)
loss = (dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)) * sp.calculate_BSpline_2D(x.detach(),
                                                                                                                                t.detach(), 
                                                                                                                                degree_1, 
                                                                                                                                degree_2, 
                                                                                                                                coef_float, 
                                                                                                                                coef_float_2)

# loss (2)
loss = torch.trapezoid((dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)) * sp.calculate_BSpline_2D(x.detach(),
                                                                                                                                t.detach(), 
                                                                                                                                degree_1, 
                                                                                                                                degree_2, 
                                                                                                                                coef_float, 
                                                                                                                                coef_float_2), dx = 0.01)

# loss (2) całka w dwóch kierunkach?
loss = torch.trapezoid(torch.trapezoid((dfdt(pinn, x, t, order=1) - eps_interior*dfdt(pinn, x, t, order=2)-eps_interior*dfdx(pinn, x, t, order=2)) * sp.calculate_BSpline_2D(x.detach(),
                                                                                                                                t.detach(), 
                                                                                                                                degree_1, 
                                                                                                                                degree_2, 
                                                                                                                                coef_float, 
                                                                                                                                coef_float_2), dx = 0.01), dx = 0.01)

# loss (1)

class B_Splines:

  def __init__(self, knot_vector: list):
    self.knot_vector = knot_vector

  # 1D case
  def calculate_BSpline_1D(self, x: torch.Tensor, coef: torch.Tensor, degree: int) -> torch.Tensor:
    return spi.BSpline(self.knot_vector, coef, degree)(x)

  def calculate_BSpline_1D_deriv(self, x: torch.Tensor, coef: torch.Tensor, degree: int, order: int) -> torch.Tensor:
    return spi.splev(x, (self.knot_vector, coef, degree), order)

  # 2D case
  def calculate_BSpline_2D(self, 
                           x1: torch.Tensor,
                           x2: torch.Tensor, 
                           degree_1: int, 
                           degree_2: int,
                           coef_1: torch.Tensor,
                           coef_2: torch.Tensor) -> torch.Tensor:

    first_spline = torch.Tensor(self.calculate_BSpline_1D(x1, coef_1, degree_1))
    second_spline = torch.Tensor(self.calculate_BSpline_1D(x2, coef_2, degree_2))
    return torch.mul(first_spline, second_spline)

      # 2D case
  def calculate_BSpline_2D_deriv_x(self, 
                           x: torch.Tensor,
                           t: torch.Tensor, 
                           degree_x: int, 
                           degree_t: int,
                           coef_x: torch.Tensor,
                           coef_t: torch.Tensor,
                           order: int) -> torch.Tensor:

    first_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(x, coef_x, degree_x, 1))
    second_spline = torch.Tensor(self.calculate_BSpline_1D(t, coef_t, degree_t))
    return torch.mul(first_spline, second_spline)

  def calculate_BSpline_2D_deriv_t(self, 
                           x: torch.Tensor,
                           t: torch.Tensor, 
                           degree_x: int, 
                           degree_t: int,
                           coef_x: torch.Tensor,
                           coef_t: torch.Tensor,
                           order: int) -> torch.Tensor:

    first_spline = torch.Tensor(self.calculate_BSpline_1D(x, coef_x, degree_x))
    second_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(t, coef_t, degree_t, 1))
    return torch.mul(first_spline, second_spline)

v = sp.calculate_BSpline_2D(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2)
v_deriv_x = sp.calculate_BSpline_2D_deriv_x(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
v_deriv_t = sp.calculate_BSpline_2D_deriv_t(x.detach(), t.detach(), degree_1, degree_2, coef_float, coef_float_2, order=1)
loss = torch.trapezoid(torch.trapezoid(
    
    dfdx(pinn, x, t, order=1) * v
    + eps_interior*dfdx(pinn, x, t, order=2) * v_deriv_x
    + eps_interior*dfdt(pinn, x, t, order=2) * v_deriv_t
    
    , dx = 0.01), dx = 0.01)
