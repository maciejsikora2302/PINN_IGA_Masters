import numpy as np
import scipy.interpolate as spi
import torch
from general_parameters import GeneralParameters
from torch.autograd import Variable
from copy import deepcopy


class B_Splines:

   def __init__(self, knot_vector: torch.Tensor, degree: int):
      self.knot_vector = knot_vector
      self.coefs = Variable(10.0 * torch.rand(len(knot_vector) - degree - 1), requires_grad=True) # We want to differentiate function wrt BSplines coefficient
      self.degree = degree


   def calculate_BSpline_1D(self, x: torch.Tensor) -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 1D splines basis functions
      """

      n = len(self.knot_vector) - self.degree - 1
      assert len(self.coefs) >= n

      def _B(x: torch.Tensor, k: int, i: int, t: torch.Tensor) -> torch.Tensor:
         """
         Function calculates i-th spline function with degree equals to k
         """
         if k == 0:
            first_condition = t[i] <= x
            second_condition = t[i+1] > x

            mask = torch.logical_and(first_condition, second_condition)
            return 1.0 if torch.all(mask) else 0.0
         if t[i+k] == t[i]:
            c1 = 0.0
         else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * _B(x, k-1, i, t)
         if t[i+k+1] == t[i+1]:
            c2 = 0.0
         else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _B(x, k-1, i+1, t)
         return c1 + c2

      basis_functions = torch.stack([_B(x, self.degree, basis_function_idx, self.knot_vector) for basis_function_idx in range(n)])

      return torch.matmul(self.coefs, basis_functions)
   
   def calculate_BSpline_2D(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 2D splines basis functions
      """

      spline_x = self.calculate_BSpline1D(x)
      spline_t = self.calculate_BSpline1D(t)

      return spline_x * spline_t
   
   def calculate_BSpline_1D_deriv_dx(self, x: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivative of BSpline function in 1D case wrt. x
      """
      x = x.detach()
      knot_vector = deepcopy(self.knot_vector)
      coefs = deepcopy(self.coefs)
      tck = (
            knot_vector.detach(),
            coefs.detach(),
            self.degree
         )
      
      return torch.Tensor(spi.splev(x, tck, der=1))
   
   def calculate_BSpline_2D_deriv_dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt x
      """

      return self.calculate_BSpline_1D_deriv_dx(x) * self.calculate_BSpline_1D(t)
   
   def calculate_BSpline_2D_deriv_dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt t
      """

      return self.calculate_BSpline_1D(x) * self.calculate_BSpline_1D_deriv_dx(t)
   
   def calculate_BSpline_2D_deriv_dxdt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of second order derivative of BSpline function in 2D case wrt x and t. Please
      note that this the same what derivative of BSpline function in 2D case wrt t and y respectively.
      The order of variables doesn't matter.
      """

      return self.calculate_BSpline_1D_deriv_dx(x) * self.calculate_BSpline_1D_deriv_dx(t)

      

# class B_Splines1D(torch.autograd.Function):

  
#   def calculate_BSpline_1D(x: torch.Tensor):
#      return spi.splev(x, (np.linspace(0, 1), [1,1,1,1,1], 3), der=0)
  
#   @staticmethod
#   def forward(ctx, x: torch.Tensor):
#       """
#       In the forward pass we receive a Tensor containing the input and return
#       a Tensor containing the output. ctx is a context object that can be used
#       to stash information for backward computation. You can cache arbitrary
#       objects for use in the backward pass using the ctx.save_for_backward method.
#       """
#       ctx.save_for_backward(x)
#       dtype = torch.float
#       device = torch.device("cpu")
#       # a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
#       # b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
#       # c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
#       # d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)
#       return torch.Tensor(spi.splev(x, (torch.Tensor(np.linspace(0,1,10)), [1,1,1,1], 3), der=0))

#   @staticmethod
#   def backward(ctx, grad_output):
#       """
#       In the backward pass we receive a Tensor containing the gradient of the loss
#       with respect to the output, and we need to compute the gradient of the loss
#       with respect to the input.
#       """
#       x, = ctx.saved_tensors
#       # knot_vector = torch.Tensor(np.linspace(0,1,1000))
#       # dtype = torch.float
#       # device = torch.device("cpu")
#       # a = torch.full((), 1, device=device, dtype=dtype, requires_grad=True)
#       # b = torch.full((), 1, device=device, dtype=dtype, requires_grad=True)
#       # c = torch.full((), 1, device=device, dtype=dtype, requires_grad=True)
#       # d = torch.full((), 1, device=device, dtype=dtype, requires_grad=True)
#       return torch.Tensor(grad_output * spi.splev(x, (torch.Tensor(np.linspace(0,1,10)), [1,1,1,1], 3), der=1))
  


# class B_Splines:


#   # 2D case
#   def calculate_BSpline_2D(self, 
#                            x1: torch.Tensor,
#                            x2: torch.Tensor, 
#                            degree_1: int, 
#                            degree_2: int,
#                            coef_1: torch.Tensor,
#                            coef_2: torch.Tensor) -> torch.Tensor:

#     first_spline = torch.Tensor(self.calculate_BSpline_1D(x1, coef_1, degree_1)).cuda()
#     second_spline = torch.Tensor(self.calculate_BSpline_1D(x2, coef_2, degree_2)).cuda()

#     first_spline = first_spline.unsqueeze(0).cuda()
#     second_spline = second_spline.unsqueeze(1).cuda()

#     return torch.mul(first_spline, second_spline)

#       # 2D case
#   def calculate_BSpline_2D_deriv_x(self, 
#                            x: torch.Tensor,
#                            t: torch.Tensor, 
#                            degree_x: int, 
#                            degree_t: int,
#                            coef_x: torch.Tensor,
#                            coef_t: torch.Tensor,
#                            order: int) -> torch.Tensor:

#     first_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(x, coef_x, degree_x, 1)).cuda()
#     second_spline = torch.Tensor(self.calculate_BSpline_1D(t, coef_t, degree_t)).cuda()

#     first_spline = first_spline.unsqueeze(0).cuda()
#     second_spline = second_spline.unsqueeze(1).cuda()
    
#     return torch.mul(first_spline, second_spline).cuda()

#   def calculate_BSpline_2D_deriv_t(self, 
#                            x: torch.Tensor,
#                            t: torch.Tensor, 
#                            degree_x: int, 
#                            degree_t: int,
#                            coef_x: torch.Tensor,
#                            coef_t: torch.Tensor,
#                            order: int) -> torch.Tensor:

#     first_spline = torch.Tensor(self.calculate_BSpline_1D(x, coef_x, degree_x)).cuda()
#     second_spline = torch.Tensor(self.calculate_BSpline_1D_deriv(t, coef_t, degree_t, 1)).cuda()

#     first_spline = first_spline.unsqueeze(0).cuda()
#     second_spline = second_spline.unsqueeze(1).cuda()

#     return torch.mul(first_spline, second_spline).cuda()
