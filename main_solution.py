
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import torch.nn as nn

from PINN import PINN
from general_parameters import general_parameters
from utils import initial_condition, running_average
from plotting import plot_solution, plot_color
from B_Splines import B_Splines
from loss_functions import compute_loss
from NN_tools import train_model

# do sprawdzenia potem co to robi
from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=float)
parser.add_argument("--total_time", type=float)
parser.add_argument("--n_points_x", type=int)
parser.add_argument("--n_points_t", type=int)
parser.add_argument("--n_points_init", type=int)
parser.add_argument("--weight_interior", type=float)
parser.add_argument("--weight_initial", type=float)
parser.add_argument("--weight_boundary", type=float)
parser.add_argument("--layers", type=int)
parser.add_argument("--neurons_per_layer", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)
args = parser.parse_args()

general_parameters.length = args.length if args.length is not None else general_parameters.length
general_parameters.total_time = args.total_time if args.total_time is not None else general_parameters.total_time
general_parameters.n_points_x = args.n_points_x if args.n_points_x is not None else general_parameters.n_points_x
general_parameters.n_points_t = args.n_points_t if args.n_points_t is not None else general_parameters.n_points_t
general_parameters.n_points_init = args.n_points_init if args.n_points_init is not None else general_parameters.n_points_init
general_parameters.weight_interior = args.weight_interior if args.weight_interior is not None else general_parameters.weight_interior
general_parameters.weight_initial = args.weight_initial if args.weight_initial is not None else general_parameters.weight_initial
general_parameters.weight_boundary = args.weight_boundary if args.weight_boundary is not None else general_parameters.weight_boundary
general_parameters.layers = args.layers if args.layers is not None else general_parameters.layers
general_parameters.neurons_per_layer = args.neurons_per_layer if args.neurons_per_layer is not None else general_parameters.neurons_per_layer
general_parameters.epochs = args.epochs if args.epochs is not None else general_parameters.epochs
general_parameters.learning_rate = args.learning_rate if args.learning_rate is not None else general_parameters.learning_rate

LENGTH = general_parameters.length
TOTAL_TIME = general_parameters.total_time
N_POINTS_X = general_parameters.n_points_x
N_POINTS_T = general_parameters.n_points_t
N_POINTS_INIT = general_parameters.n_points_init
WEIGHT_INTERIOR = general_parameters.weight_interior
WEIGHT_INITIAL = general_parameters.weight_initial
WEIGHT_BOUNDARY = general_parameters.weight_boundary
LAYERS = general_parameters.layers
NEURONS_PER_LAYER = general_parameters.neurons_per_layer
EPOCHS = general_parameters.epochs
LEARNING_RATE = general_parameters.learning_rate

if __name__ == "__main__":

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

