from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from PINN import PINN

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


LENGTH = 1.
TOTAL_TIME = 1.
N_POINTS_X = 150
N_POINTS_T = 150
N_POINTS_INIT = 300
WEIGHT_INTERIOR = 0.5
WEIGHT_INITIAL = 150.0
WEIGHT_BOUNDARY = 1.0
LAYERS = 2
NEURONS_PER_LAYER = 60
EPOCHS = 50_000
LEARNING_RATE = 0.0025




x_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=N_POINTS_T, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

x = grids[0].flatten().reshape(-1, 1).to(device)
t = grids[1].flatten().reshape(-1, 1).to(device)

x_init = torch.linspace(0.0, 1.0, steps=N_POINTS_INIT)
# x_init = 0.5*((x_init-0.5*LENGTH)*2)**3 + 0.5
x_init = x_init*LENGTH
u_init = initial_condition(x_init)

fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Initial condition points")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.scatter(x_init, u_init, s=2)


pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=False, act=nn.Tanh()).to(device)
# assert check_gradient(nn_approximator, x, t)

compute_loss(pinn, x=x, t=t)

# train the PINN
loss_fn = partial(compute_loss, x=x, t=t, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY)
pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)




losses = compute_loss(pinn.to(device), x=x, t=t, verbose=True)
print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
print(f'Bondary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')




average_loss = running_average(loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (runnig average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)




z = f(pinn.to(device), x, t)
color = plot_color(z.cpu(), x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T)




# plt.plot(x_init, u_init, label="Initial condition")
# plt.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
# plt.legend()

pinn_init = f(pinn.cpu(), x_init.reshape(-1, 1), torch.zeros_like(x_init).reshape(-1,1))
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Initial condition difference")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.plot(x_init, u_init, label="Initial condition")
ax.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
ax.legend()




# from IPython.display import HTML
# ani = plot_solution(pinn_trained.cpu(), x.cpu(), t.cpu())
# HTML(ani.to_html5_video())




# plt.plot(x_init, u_init, label="Initial condition")
# plt.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
# plt.legend()

pinn_init = f(pinn.cpu(), torch.zeros_like(x_init).reshape(-1,1)+0.5, x_init.reshape(-1, 1))
fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
ax.set_title("Solution profile")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
ax.legend()

