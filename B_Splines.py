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
   
   def calculate_BSpline_1D_deriv_dx(self, x: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of derivative of BSpline function in 1D case wrt. x
      """


      def _de_Boor_derivative(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, p: int):
         """
         Evaluates first order derivative of a linear combination of B-Splines basis functions

         Args
         ----
         x: position
         t: array of knot positions, needs to be padded as described above
         c: array of control points
         p: degree of B-spline
         """

         result = torch.zeros_like(x)

         for idx, elem in enumerate(x):

            k = torch.searchsorted(t, x[idx], side='right') - 1

            q = [p * (c[j+k-p+1] - c[j+k-p]) / (t[j+k+1] - t[j+k-p+1]) for j in range(0, p)]

            for r in range(1, p):
               for j in range(p-1, r-1, -1):
                     right = j+1+k-r
                     left = j+k-(p-1)
                     alpha = (elem - t[left]) / (t[right] - t[left])
                     q[j] = (1.0 - alpha) * q[j-1] + alpha * q[j]
            result[idx] = q[p-1]

         return result
      
      if mode == 'NN':
         x = x.cpu().detach()
         knot_vector = deepcopy(self.knot_vector)
         coefs = deepcopy(self.coefs)

         #repeat and add first and last element of knot_vector twice
         knot_vector = torch.cat((knot_vector[0].repeat(2), knot_vector, knot_vector[-1].repeat(2)))

         tck = (
               knot_vector.detach(),
               coefs.detach(),
               self.degree
            )
         
         return torch.Tensor(spi.splev(x, tck, der=1))
      
      elif mode == 'Adam':
         return torch.Tensor(_de_Boor_derivative(x, self.knot_vector, self.coefs, self.degree))
   
   def calculate_BSpline_1D_deriv_dxdx(self, x:torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second derivative of BSpline function in 1D case wrt. x
      """

      if mode == 'NN':
         x = x.cpu().detach()
         knot_vector = deepcopy(self.knot_vector)
         coefs = deepcopy(self.coefs)
         tck = (
               knot_vector.detach(),
               coefs.detach(),
               self.degree
            )
         return torch.Tensor(spi.splev(x, tck, der=2))
      
      elif mode == 'Adam':
         f_dx = self.calculate_BSpline_1D_deriv_dx(x, mode=mode)
         f_dxdx = self.calculate_BSpline_1D_deriv_dx(f_dx, mode=mode)

         return f_dxdx
      
      
   
   def calculate_BSpline_2D_deriv_dx(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      return torch.outer(self.calculate_BSpline_1D_deriv_dx(x, mode=mode).cpu(), self.calculate_BSpline_1D(t, mode=mode).cpu())
   
   def calculate_BSpline_2D_deriv_dxdx(self, x: torch.Tensor, t: torch.Tensor, mode: str = None) -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      spline_1D_deriv_dx = self.calculate_BSpline_1D_deriv_dx(x, mode=mode).cpu()
      spline_1D_deriv_dxdx = self.calculate_BSpline_1D_deriv_dx(spline_1D_deriv_dx, mode=mode).cpu()
      spline_1D_t = self.calculate_BSpline_1D(t, mode=mode).cpu()

      return torch.outer(spline_1D_deriv_dxdx, spline_1D_t).cpu()


   def calculate_BSpline_2D_deriv_dtdt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()

      spline_1D_deriv_dt = self.calculate_BSpline_1D_deriv_dx(x, mode=mode).cpu()
      spline_1D_deriv_dtdt = self.calculate_BSpline_1D_deriv_dx(spline_2D_deriv_dt, mode=mode).cpu()
      spline_1D_x = self.calculate_BSpline_1D(x, mode=mode).cpu()

      return torch.outer(spline_1D_deriv_dtdt, spline_1D_x).cpu()

   def calculate_BSpline_2D_deriv_dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()
      
      return torch.outer(self.calculate_BSpline_1D(x).cpu(), self.calculate_BSpline_1D_deriv_dx(t).cpu())
   
   def calculate_BSpline_2D_deriv_dxdt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second order derivative of BSpline function in 2D case wrt x and t. Please
      note that this the same what derivative of BSpline function in 2D case wrt t and y respectively.
      The order of variables doesn't matter.
      """

      return torch.outer(self.calculate_BSpline_1D_deriv_dx(x,mode=mode), self.calculate_BSpline_1D_deriv_dx(t, mode=mode))
   
   def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:

      if self.dims == 1:
         return self.calculate_BSpline_1D(x)
      elif self.dims == 2:
         return self.calculate_BSpline_2D(x, t)
