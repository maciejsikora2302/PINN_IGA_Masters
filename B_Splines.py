
import scipy.interpolate as spi
import torch

class B_Splines:

  def __init__(self, knot_vector: list):
    self.knot_vector = knot_vector

  # 1D case
  def calculate_BSpline_1D(self, x: torch.Tensor, coef: torch.Tensor, degree: int, order: int = None) -> torch.Tensor:
    return spi.BSpline(self.knot_vector, coef, degree)(x.cpu())

  def calculate_BSpline_1D_deriv(self, x: torch.Tensor, coef: torch.Tensor, degree: int, order: int) -> torch.Tensor:
    return spi.splev(x.cpu(), (self.knot_vector, coef, degree), order)

  # 2D case
  def calculate_BSpline_2D(self, 
                           x1: torch.Tensor,
                           x2: torch.Tensor, 
                           degree_1: int, 
                           degree_2: int,
                           coef_1: torch.Tensor,
                           coef_2: torch.Tensor) -> torch.Tensor:

    first_spline = torch.Tensor(self.calculate_BSpline_1D(x1, coef_1, degree_1)).cuda()
    second_spline = torch.Tensor(self.calculate_BSpline_1D(x2, coef_2, degree_2)).cuda()

    first_spline = first_spline.unsqueeze(0).cuda()
    second_spline = second_spline.unsqueeze(1).cuda()

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

    first_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(x, coef_x, degree_x, 1)).cuda()
    second_spline = torch.Tensor(self.calculate_BSpline_1D(t, coef_t, degree_t)).cuda()

    first_spline = first_spline.unsqueeze(0).cuda()
    second_spline = second_spline.unsqueeze(1).cuda()
    
    return torch.mul(first_spline, second_spline).cuda()

  def calculate_BSpline_2D_deriv_t(self, 
                           x: torch.Tensor,
                           t: torch.Tensor, 
                           degree_x: int, 
                           degree_t: int,
                           coef_x: torch.Tensor,
                           coef_t: torch.Tensor,
                           order: int) -> torch.Tensor:

    first_spline = torch.Tensor(self.calculate_BSpline_1D(x, coef_x, degree_x)).cuda()
    second_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(t, coef_t, degree_t, 1)).cuda()

    first_spline = first_spline.unsqueeze(0).cuda()
    second_spline = second_spline.unsqueeze(1).cuda()

    return torch.mul(first_spline, second_spline).cuda()
