import scipy.interpolate as spi
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

class B_Splines(torch.nn.Module):

   def __init__(self, knot_vector: torch.Tensor, degree: int, coefs: torch.Tensor = None, dims: int = 1):

      super().__init__()
      self.knot_vector = knot_vector
      self.degree = degree
      self.n_coeff = len(self.knot_vector) - self.degree - 1 if dims == 1 else (len(self.knot_vector) - self.degree - 1)**2
      self.coefs = torch.nn.Parameter(torch.rand(self.n_coeff) if coefs is None else coefs)
      # self.coefs = torch.nn.Parameter(torch.ones(self.n_coeff)) if coefs is None else coefs
      self.dims = dims
      self.losses = []

   def _get_basis_functions_1D(self, x: torch.Tensor, order: int = 0) -> torch.Tensor:
      """
      Function returns tensor of B-spline basis functions calculated using scipy framework. This method will be helpful to
      speed up calculations of framework during training PINNs to estimate splines coefficient.
      """

      n_coefs = self.n_coeff

      basis_functions = []

      for idx in range(n_coefs):

         coefs = torch.zeros(n_coefs)
         coefs[idx] = 1.0
         tck = (
            self.knot_vector.to(device_cpu),
            coefs.to(device_cpu),
            self.degree
         )
         # tck = (
         #    self.knot_vector.to(device_cpu).detach(),
         #    coefs.to(device_cpu).detach(),
         #    self.degree
         # )

         BS = spi.splev(x.to(device_cpu), tck, der=order, ext=0)
         basis_functions.append(BS)

      return torch.Tensor(basis_functions).T.to(device)
   
   def _get_basis_functions_2D(self, x: torch.Tensor, t: torch.Tensor, order: int = 0) -> torch.Tensor:
      """
      Function returns tensor of B-spline basis functions calculated using scipy framework. This method will be helpful to
      speed up calculations of framework during training PINNs to estimate splines coefficient.
      """

      raise NotImplementedError("Not implemented yet")

      n_coefs = (len(self.knot_vector) - self.degree - 1)**2

      basis_functions = []

      for idx in range(n_coefs):

         coefs = torch.zeros(n_coefs)
         coefs[idx] = 1.0
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )
         BS = spi.bisplev(x.to(device_cpu).detach(), t.to(device_cpu).detach(), tck)
         basis_functions.append(BS)

      return torch.Tensor(basis_functions).T.to(device)

   def _B(self, x: torch.Tensor, k: int, i: int, t: torch.Tensor) -> torch.Tensor:
      """
      Function calculates i-th spline function with degree equals to k
      """
      if k == 0:
         first_condition = t[i] <= x
         second_condition = t[i+1] > x
         
         mask = torch.logical_and(first_condition, second_condition)

         if i == self.degree:
            mask[0] = True

         if i == self.n_coeff-1:
            mask[-1] = True
         
         return mask.float()
      
      if t[i+k] == t[i]:
         c1 = torch.zeros_like(x)
      else:
         c1 = (x - t[i])/(t[i+k] - t[i]) * self._B(x, k-1, i, t)

      if t[i+k+1] == t[i+1]:

         c2 = torch.zeros_like(x)
      else:
         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * self._B(x, k-1, i+1, t)
      
      return c1 + c2

   def calculate_BSpline_1D(self, x: torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None, return_bs_stacked: bool = False) -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 1D splines basis functions
      """
      n = len(self.knot_vector) - self.degree - 1
      
      coefs = self.coefs if coefs is None else coefs

      x = x.flatten()
      
      
      if mode == 'Adam':
         basis_functions = torch.stack([self._B(x, self.degree, basis_function_idx, self.knot_vector) for basis_function_idx in range(n)])
         
         return torch.matmul(coefs.to(device), basis_functions.to(device)) if not return_bs_stacked else basis_functions
      
      else:

         tck = (self.knot_vector.detach(), coefs.detach(), self.degree)
         
         return torch.Tensor(spi.splev(x.to(device_cpu).detach(), tck, der=0)).to(device)
   
   def calculate_BSpline_2D(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Funtion calculates value of a linear combination of 2D splines basis functions
      """

      if mode == 'NN':

         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            self.coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )

         spline_2d = spi.bisplev(
            x.flatten().cpu().detach().numpy(),
            t.flatten().cpu().detach().numpy(),
            tck
         )
      
      elif mode == 'Adam':
         n_coeff_sqrt = int(math.sqrt(self.n_coeff))
         
         splines_x = self.calculate_BSpline_1D(x, mode=mode, return_bs_stacked=True)
         splines_t = self.calculate_BSpline_1D(t, mode=mode, return_bs_stacked=True)

         spline_2d = splines_x.T @ self.coefs.reshape(n_coeff_sqrt, n_coeff_sqrt) @ splines_t
      
      return torch.Tensor(spline_2d)


   def calculate_BSpline_1D_deriv_dx(self, x: torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None, return_bs_stacked: bool = False) -> torch.Tensor:
      """
      Function returns value of derivative of BSpline function in 1D case wrt. x
      """

      coefs = self.coefs if coefs is None else coefs

      if mode == 'NN':
         # x = x.to(device_cpu)
         knot_vector = self.knot_vector
         x = x.to(device_cpu).detach()
         knot_vector = self.knot_vector

         #repeat and add first and last element of knot_vector twice
         # knot_vector = torch.cat((knot_vector[0].repeat(2), knot_vector, knot_vector[-1].repeat(2)))

         # tck = (
         #       knot_vector,
         #       coefs,
         #       self.degree
         #    )
         tck = (
               knot_vector,
               coefs.detach(),
               self.degree
            )
         
         return torch.Tensor(spi.splev(x, tck, der=1)).to(device)
      
      elif mode == 'Adam':
         x = x.flatten().to(device_cpu)
         
         p = self.degree
         u = self.knot_vector
         n = len(u) - p - 1
         B_ip = lambda i: self._B(x, p-1, i, u)

         basis_functions = torch.stack([p * (B_ip(i) / (u[i+p] - u[i]) - B_ip(i+1) / (u[i+p+1] - u[i+1])) for i in range(n)])

         return torch.matmul(coefs, basis_functions) if not return_bs_stacked else basis_functions
   
   def calculate_BSpline_1D_deriv_dxdx(self, x:torch.Tensor, mode: str = 'NN', coefs: torch.Tensor = None, return_bs_stacked: bool = False) -> torch.Tensor:
      """
      Function returns value of second derivative of BSpline function in 1D case wrt. x
      """

      coefs = self.coefs if coefs is None else coefs

      if mode == 'NN':
         x = x.to(device_cpu).detach()
         tck = (
               self.knot_vector.detach(),
               coefs.detach(),
               self.degree
            )
         # x = x.to(device_cpu).detach()
         # knot_vector = self.knot_vector.clone().detach()
         # tck = (
         #       knot_vector,
         #       coefs.detach(),
         #       self.degree
         #    )
         return torch.Tensor(spi.splev(x, tck, der=2))
      
      elif mode == 'Adam':

         p = self.degree
         u = self.knot_vector
         n = len(u) - p - 1
         B_ip = lambda i: self._B(x, self.degree - 2, i, self.knot_vector)

         basis_functions_dxdx = torch.stack(
            [p * (p - 1) * ((B_ip(i) / (u[i + p - 1] - u[i]) - B_ip(i+1) / (u[i + p] - u[i + 1])) / (u[i + p] - u[i]) - (B_ip(i+1)/(u[i+p] - u[i+1]) - B_ip(i+2)/(u[i+p+1] - u[i+2])) / (u[i+p+1] - u[i+1]) ) for i in range(n)]
            )
         
         return torch.matmul(coefs, basis_functions_dxdx) if not return_bs_stacked else basis_functions_dxdx
      
      
   
   def calculate_BSpline_2D_deriv_dx(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt x
      """
      x = x.flatten()
      t = t.flatten()

      if mode == 'NN':
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            self.coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )

         spline_2d = spi.bisplev(
            x.to(device_cpu),
            t.to(device_cpu),
            tck,
            dx=1
         )
      
      elif mode == 'Adam':
         
         basis_functions_dx = self.calculate_BSpline_1D_deriv_dx(x, mode=mode, return_bs_stacked=True)
         basis_functions_t = self.calculate_BSpline_1D(t, mode=mode, return_bs_stacked=True)
      
         coefs_dim_1 = basis_functions_dx.shape[0]
         coefs_dim_2 = basis_functions_t.shape[0]

         spline_2d = basis_functions_dx.T @ self.coefs.reshape(coefs_dim_1, coefs_dim_2) @ basis_functions_t


      return torch.Tensor(spline_2d)

   
   def calculate_BSpline_2D_deriv_dxdx(self, x: torch.Tensor, t: torch.Tensor, mode: str = None) -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt x
      """
      
      x = x.flatten()
      t = t.flatten()

      if mode == 'NN':
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            self.coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )

         spline_2d = spi.bisplev(
            x.to(device_cpu),
            t.to(device_cpu),
            tck,
            dx=2
         )
      
      elif mode == 'Adam':

         basis_functions_dxdx = self.calculate_BSpline_1D_deriv_dxdx(x, mode=mode, return_bs_stacked=True)
         basis_functions_t = self.calculate_BSpline_1D(t, mode=mode, return_bs_stacked=True)


         coefs_dim_1 = basis_functions_dxdx.shape[0]
         coefs_dim_2 = basis_functions_t.shape[0]

         spline_2d = basis_functions_dxdx.T @ self.coefs.reshape(coefs_dim_1, coefs_dim_2) @ basis_functions_t


      return torch.Tensor(spline_2d)


   def calculate_BSpline_2D_deriv_dtdt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'NN') -> torch.Tensor:
      """
      Function returns value of second derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()

      if mode == 'NN':
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            self.coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )

         spline_2d = spi.bisplev(
            x.to(device_cpu),
            t.to(device_cpu),
            tck,
            dy=2
         )
      
      elif mode == 'Adam':

         basis_functions = self.calculate_BSpline_1D(x, mode=mode, return_bs_stacked=True)
         basis_functions_dtdt = self.calculate_BSpline_1D_deriv_dxdx(t, mode=mode, return_bs_stacked=True)


         coefs_dim_1 = basis_functions.shape[0]
         coefs_dim_2 = basis_functions_dtdt.shape[0]

         spline_2d = basis_functions.T @ self.coefs.reshape(coefs_dim_1, coefs_dim_2) @ basis_functions_dtdt


      return torch.Tensor(spline_2d)

   def calculate_BSpline_2D_deriv_dt(self, x: torch.Tensor, t: torch.Tensor, mode: str = 'Adam') -> torch.Tensor:
      """
      Function returns value of derivtive of BSpline function in 2D case wrt t
      """
      x = x.flatten()
      t = t.flatten()

      if mode == 'NN':
         tck = (
            self.knot_vector.to(device_cpu).detach(),
            self.knot_vector.to(device_cpu).detach(),
            self.coefs.to(device_cpu).detach(),
            self.degree,
            self.degree
         )

         spline_2d = spi.bisplev(
            x.to(device_cpu),
            t.to(device_cpu),
            tck,
            dy=1
         )
      
      elif mode == 'Adam':
         
         basis_functions_x = self.calculate_BSpline_1D(x, mode=mode, return_bs_stacked=True)
         basis_functions_dt = self.calculate_BSpline_1D_deriv_dx(t, mode=mode, return_bs_stacked=True)
      
         coefs_dim_1 = basis_functions_x.shape[0]
         coefs_dim_2 = basis_functions_dt.shape[0]

         spline_2d = basis_functions_x.T @ self.coefs.reshape(coefs_dim_1, coefs_dim_2) @ basis_functions_dt

      return torch.Tensor(spline_2d)
   
   def forward(self, x: torch.Tensor, mode:str, t: torch.Tensor = None) -> torch.Tensor:

      # x_copy = x.clone().detach()
      # x_copy.requires_grad = True

      if self.dims == 1:
         return self.calculate_BSpline_1D(x, mode=mode)
      elif self.dims == 2:
         return self.calculate_BSpline_2D(x, t, mode=mode)
