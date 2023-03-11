import scipy.interpolate as spi
import torch
from copy import deepcopy


class B_Splines(torch.nn.Module):

   def __init__(self, knot_vector: torch.Tensor, degree: int, coefs: torch.Tensor = None, dims: int = 1):

      super().__init__()
      self.knot_vector = knot_vector
      # self.coefs = Variable(10.0 * torch.rand(len(knot_vector) - degree - 1), requires_grad=True) if coefs is None else coefs # We want to differentiate function wrt BSplines coefficient
      self.degree = degree
      self.coefs = torch.nn.Parameter(10.0 * torch.rand(len(knot_vector) - degree - 1) if coefs is None else coefs)
      self.dims = dims
      self.losses = []

      self.saved_splines = {} # x, k, i, t -> Tensor


   def calculate_BSpline_1D(self, x: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 1D splines basis functions
      """
      n = len(self.knot_vector) - self.degree - 1
      assert len(self.coefs) >= n

      if self.dims == 1:
         x = x.flatten().cuda()
      elif self.dims == 2:
         x = x.flatten()
      
      if mode == 'adam':

         def _B(x: torch.Tensor, k: int, i: int, t: torch.Tensor) -> torch.Tensor:
            """
            Function calculates i-th spline function with degree equals to k
            """
            if (x, k, i, t) in self.saved_splines.keys():
               return self.saved_splines[(x,k,i,t)]
            else:
               if k == 0:
                  first_condition = t[i] <= x
                  second_condition = t[i+1] > x

                  mask = torch.logical_and(first_condition, second_condition)
                  return mask
               if t[i+k] == t[i]:
                  c1 = torch.zeros_like(x)
               else:
                  c1 = (x - t[i])/(t[i+k] - t[i]) * _B(x, k-1, i, t)
               if t[i+k+1] == t[i+1]:
                  c2 = torch.zeros_like(x)
               else:
                  c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _B(x, k-1, i+1, t)
               self.saved_splines[(x,k,i,t)] = c1 + c2
               return c1 + c2

         basis_functions = torch.stack([_B(x, self.degree, basis_function_idx, self.knot_vector) for basis_function_idx in range(n)])

         return torch.matmul(self.coefs.cuda(), basis_functions)
      
      else:

         tck = (self.knot_vector.detach(), self.coefs.detach(), self.degree)

         if self.dims == 1:
            return torch.Tensor(spi.splev(x.cpu().detach(), tck, der=0)).cuda()
         elif self.dims == 2:
            return torch.Tensor(spi.splev(x.cpu().detach(), tck, der=0))
   
   def calculate_BSpline_2D(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 2D splines basis functions
      """

      # x = x.cuda()
      # t = t.cuda()

      spline_x = self.calculate_BSpline_1D(x, mode=mode)
      spline_t = self.calculate_BSpline_1D(t, mode=mode)

      return torch.outer(spline_x, spline_t)
   
   def calculate_BSpline_1D_deriv_dx(self, x: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivative of BSpline function in 1D case wrt. x
      """
      x = x.cpu().detach()
      knot_vector = deepcopy(self.knot_vector)
      coefs = deepcopy(self.coefs)
      tck = (
            knot_vector.detach(),
            coefs.detach(),
            self.degree
         )
      
      return torch.Tensor(spi.splev(x, tck, der=1))
   
   def calculate_BSpline_1D_deriv_dxdx(self, x:torch.Tensor) -> torch.Tensor:
      """
      Function returns value of second derivative of BSpline function in 1D case wrt. x
      """
      x = x.cpu().detach()
      knot_vector = deepcopy(self.knot_vector)
      coefs = deepcopy(self.coefs)
      tck = (
            knot_vector.detach(),
            coefs.detach(),
            self.degree
         )
      
      return torch.Tensor(spi.splev(x, tck, der=2))
   
   def calculate_BSpline_2D_deriv_dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      return torch.outer(self.calculate_BSpline_1D_deriv_dx(x).cpu(), self.calculate_BSpline_1D(t).cpu())
   
   def calculate_BSpline_2D_deriv_dxdx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      spline_2D_deriv_dx = self.calculate_BSpline_2D_deriv_dx(x, t).cpu()
      spline_2D_deriv_dxdx = self.calculate_BSpline_2D_deriv_dx(spline_2D_deriv_dx).cpu()
      spline_2D_t = self.calculate_BSpline_1D(t).cpu()

      return torch.outer(spline_2D_deriv_dxdx, spline_2D_t).cpu()


   def calculate_BSpline_2D_deriv_dtdt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()

      spline_2D_deriv_dt = self.calculate_BSpline_2D_deriv_dt(x, t).cpu()
      spline_2D_deriv_dtdt = self.calculate_BSpline_2D_deriv_dt(spline_2D_deriv_dt).cpu()
      spline_2D_x = self.calculate_BSpline_1D(x).cpu()

      return torch.outer(spline_2D_deriv_dtdt, spline_2D_x).cpu()

   def calculate_BSpline_2D_deriv_dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()
      
      return torch.outer(self.calculate_BSpline_1D(x).cpu(), self.calculate_BSpline_1D_deriv_dx(t).cpu())
   
   def calculate_BSpline_2D_deriv_dxdt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of second order derivative of BSpline function in 2D case wrt x and t. Please
      note that this the same what derivative of BSpline function in 2D case wrt t and y respectively.
      The order of variables doesn't matter.
      """

      return torch.outer(self.calculate_BSpline_1D_deriv_dx(x), self.calculate_BSpline_1D_deriv_dx(t))
   
   def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:

      if self.dims == 1:
         return self.calculate_BSpline_1D(x)
      elif self.dims == 2:
         return self.calculate_BSpline_2D(x, t)
