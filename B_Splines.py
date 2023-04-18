import scipy.interpolate as spi
import torch
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

class B_Splines(torch.nn.Module):

   def __init__(self, knot_vector: torch.Tensor, degree: int, coefs: torch.Tensor = None, dims: int = 1, n_coeff: int = None):

      super().__init__()
      self.n_coeff = n_coeff
      self.knot_vector = knot_vector
      self.degree = degree
      # self.coefs = torch.nn.Parameter(10.0 * torch.rand(len(self.knot_vector) - self.degree - 1) if coefs is None else coefs)
      self.coefs = torch.nn.Parameter(torch.ones(len(self.knot_vector) - self.degree - 1))
      self.coefs_2 = torch.nn.Parameter(torch.ones(len(self.knot_vector) - self.degree - 1))
      self.dims = dims
      self.losses = []

      self.saved_splines = {} # x, k, i, t -> Tensor

   def _get_basis_functions_1D(self, x: torch.Tensor, order: int = 0, coefs: torch.Tensor = None) -> torch.Tensor:
      """
      Function returns tensor of B-spline basis functions calculated using scipy framework. This method will be helpful to
      speed up calculations of framework during training PINNs to estimate splines coefficient.
      """

      n_coefs = len(self.knot_vector) - self.degree - 1
      coefs = self.coefs if coefs is None else coefs

      basis_functions = []

      for idx in range(n_coefs):

         coefs = torch.zeros(n_coefs)
         coefs[idx] = 1.0
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            coefs.to(device_cpu).detach(),
            self.degree
         )

         BS = spi.splev(x.to(device_cpu).detach(), tck, der=order, ext=0)
         basis_functions.append(BS)

      return torch.Tensor(basis_functions).T.to(device)
   
   def _get_basis_functions_2D(self, x: torch.Tensor, t: torch.Tensor, order: int = 0) -> torch.Tensor:
      """
      Function returns tensor of B-spline basis functions calculated using scipy framework. This method will be helpful to
      speed up calculations of framework during training PINNs to estimate splines coefficient.
      """

      BS_x = self._get_basis_functions_1D(x, coefs=self.coefs)
      BS_t = self._get_basis_functions_1D(t, coefs=self.coefs_2)
      
      return torch.outer(BS_x, BS_t)

   def _B(self, x: torch.Tensor, k: int, i: int, t: torch.Tensor) -> torch.Tensor:
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
            c1 = (x - t[i])/(t[i+k] - t[i]) * self._B(x, k-1, i, t)
         if t[i+k+1] == t[i+1]:
            c2 = torch.zeros_like(x)
         else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * self._B(x, k-1, i+1, t)
         self.saved_splines[(x,k,i,t)] = c1 + c2
         return c1 + c2

   def calculate_BSpline_1D(self, x: torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None) -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 1D splines basis functions
      """
      n = len(self.knot_vector) - self.degree - 1
      
      coefs = self.coefs if coefs is None else coefs

      if self.dims == 1:
         x = x.flatten()
      elif self.dims == 2:
         x = x.flatten()
      
      if mode == 'Adam':
         
         basis_functions = torch.stack([self._B(x, self.degree, basis_function_idx, self.knot_vector) for basis_function_idx in range(n)])
         
         return torch.matmul(coefs.to(device), basis_functions)
      
      else:

         tck = (self.knot_vector.detach(), coefs.detach(), self.degree)
         
         if self.dims == 1:
            return torch.Tensor(spi.splev(x.to(device_cpu).detach(), tck, der=0)).to(device)
         elif self.dims == 2:
            return torch.Tensor(spi.splev(x.to(device_cpu).detach(), tck, der=0))
   
   def calculate_BSpline_2D(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 2D splines basis functions
      """

      spline_x = self.calculate_BSpline_1D(x, mode=mode, coefs=self.coefs)
      spline_t = self.calculate_BSpline_1D(t, mode=mode, coefs=self.coefs_2)

      return torch.outer(spline_x, spline_t)
   
   def _de_Boor_derivative(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, p: int):
         """
         Evaluates first order derivative of a linear combination of B-Splines basis functions

         Args
         ----
         x: position
         t: array of knot positions, needs to be padded as described above
         c: array of control points
         p: degree of B-spline
         """
         
         x = x.flatten()

         result = torch.zeros_like(x)

         for idx, elem in enumerate(x):
            try:
               k = torch.searchsorted(t, x[idx], side='left') - 1
               
               q = [p * (c[j+k-p+1] - c[j+k-p]) / (t[j+k+1] - t[j+k-p+1]) for j in range(0, p)]

               for r in range(1, p):
                  for j in range(p-1, r-1, -1):
                        right = j+1+k-r
                        left = j+k-(p-1)
                        alpha = (elem - t[left]) / (t[right] - t[left])
                        q[j] = (1.0 - alpha) * q[j-1] + alpha * q[j]
               result[idx] = q[p-1]

            except:
               result[idx] = 0
         return result


   def calculate_BSpline_1D_deriv_dx(self, x: torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None) -> torch.Tensor:
      """
      Function returns value of derivative of BSpline function in 1D case wrt. x
      """

      coefs = self.coefs if coefs is None else coefs

      
      
      if mode == 'NN':
         x = x.to(device_cpu).detach()
         knot_vector = deepcopy(self.knot_vector)

         #repeat and add first and last element of knot_vector twice
         # knot_vector = torch.cat((knot_vector[0].repeat(2), knot_vector, knot_vector[-1].repeat(2)))

         tck = (
               knot_vector.detach(),
               coefs.detach(),
               self.degree
            )
         
         return torch.Tensor(spi.splev(x, tck, der=1))
      
      elif mode == 'Adam':
         x = x.flatten().to(device_cpu)

         return torch.Tensor(self._de_Boor_derivative(x, self.knot_vector, coefs, self.degree))
   
   def calculate_BSpline_1D_deriv_dxdx(self, x:torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None) -> torch.Tensor:
      """
      Function returns value of second derivative of BSpline function in 1D case wrt. x
      """

      coefs = self.coefs if coefs is None else coefs

      if mode == 'NN':
         x = x.to(device_cpu).detach()
         knot_vector = deepcopy(self.knot_vector)
         tck = (
               knot_vector.detach(),
               coefs.detach(),
               self.degree
            )
         return torch.Tensor(spi.splev(x, tck, der=2))
      
      elif mode == 'Adam':

         n = self.n_coeff

         basis_functions = torch.stack([self._B(x, self.degree - 2, basis_function_idx + 2, self.knot_vector) for basis_function_idx in range(n-2)])
         
         p = self.degree
         t = self.knot_vector
         c = coefs

         coefs = [p*(p-1)/(t[i+p]-t[i+1]) * ( (c[i+2] - c[i+1])/(t[i+p+2] - t[i+2]) - ( c[i+1] - c[i] )/( t[i+p+1] - t[i+1] ) ) for i in range(n-2)]
         
         coefs = torch.Tensor(coefs)

         return torch.matmul(coefs, basis_functions)
      
      
   
   def calculate_BSpline_2D_deriv_dx(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      return torch.outer(
         self.calculate_BSpline_1D_deriv_dx(x, mode=mode, coefs=self.coefs).to(device_cpu),
         self.calculate_BSpline_1D(t, mode=mode, coefs=self.coefs_2).to(device_cpu)
         )
   
   def calculate_BSpline_2D_deriv_dxdx(self, x: torch.Tensor, t: torch.Tensor, mode: str = None) -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      spline_1D_deriv_dxdx = self.calculate_BSpline_1D_deriv_dx(x, mode=mode, coefs=self.coefs).to(device_cpu)
      spline_1D_t = self.calculate_BSpline_1D(t, mode=mode, coefs=self.coefs_2).to(device_cpu)

      return torch.outer(spline_1D_deriv_dxdx, spline_1D_t).to(device_cpu)


   def calculate_BSpline_2D_deriv_dtdt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()

      spline_1D_deriv_dtdt = self.calculate_BSpline_1D_deriv_dx(t, mode=mode, coefs=self.coefs_2).to(device_cpu)
      spline_1D_x = self.calculate_BSpline_1D(x, mode=mode, coefs=self.coefs).to(device_cpu)

      return torch.outer(spline_1D_deriv_dtdt, spline_1D_x).to(device_cpu)

   def calculate_BSpline_2D_deriv_dt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'Adam') -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()
      
      spline_1D_x = self.calculate_BSpline_1D(x, mode=mode, coefs=self.coefs).to(device_cpu)
      spline_1D_deriv_dt = self.calculate_BSpline_1D_deriv_dx(t, mode=mode, coefs=self.coefs_2).to(device_cpu)

      return torch.outer(spline_1D_x, spline_1D_deriv_dt).to(device_cpu)
   
   def calculate_BSpline_2D_deriv_dxdt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second order derivative of BSpline function in 2D case wrt x and t. Please
      note that this the same what derivative of BSpline function in 2D case wrt t and y respectively.
      The order of variables doesn't matter.
      """

      spline_1D_deriv_dx = self.calculate_BSpline_1D_deriv_dx(x, mode=mode, coefs=self.coefs)
      spline_1D_deriv_dt = self.calculate_BSpline_1D_deriv_dx(t, mode=mode, coefs=self.coefs_2)

      return torch.outer(spline_1D_deriv_dx, spline_1D_deriv_dt).to(device_cpu)
   
   def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:

      if self.dims == 1:
         return self.calculate_BSpline_1D(x)
      elif self.dims == 2:
         return self.calculate_BSpline_2D(x, t)
