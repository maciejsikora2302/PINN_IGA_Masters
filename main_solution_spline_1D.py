import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from functools import partial

from PINN import PINN
from general_parameters import general_parameters, logger, Color, TIMESTAMP
from utils import compute_losses_and_plot_solution
from loss_functions import iga_loss, compute_loss, initial_condition
from NN_tools import train_model
from B_Splines import B_Splines


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

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
parser.add_argument("--eps_interior", type=float)
parser.add_argument("--spline_degree", type=int)
parser.add_argument("--save", '-s', action="store_true")
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
general_parameters.eps_interior = args.eps_interior if args.eps_interior is not None else general_parameters.eps_interior
general_parameters.spline_degree = args.spline_degree if args.spline_degree is not None else general_parameters.spline_degree
general_parameters.save = args.save if args.save is not None else general_parameters.save

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
SAVE = general_parameters.save



if __name__ == "__main__":

    x_domain = [0.0, LENGTH]

    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)

    x = x_raw.flatten().reshape(-1, 1).to(device)

    x_init = torch.linspace(0.0, 1.0, steps=N_POINTS_INIT)
    # x_init = 0.5*((x_init-0.5*LENGTH)*2)**3 + 0.5
    x_init = x_init*LENGTH
    u_init = initial_condition(x_init)

    # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    # ax.set_title("Initial condition points")
    # ax.set_xlabel("x")
    # ax.set_ylabel("u")
    # ax.scatter(x_init, u_init, s=2)

    logger.info("")
    logger.info("="*80)
    logger.info("Learning parameters")
    logger.info(f"{'Length: ':<50}{Color.GREEN}{LENGTH}{Color.RESET}")
    logger.info(f"{'Total time: ':<50}{Color.GREEN}{TOTAL_TIME}{Color.RESET}")
    logger.info(f"{'Number of points in x: ':<50}{Color.GREEN}{N_POINTS_X}{Color.RESET}")
    logger.info(f"{'Number of points in initial condition: ':<50}{Color.GREEN}{N_POINTS_INIT}{Color.RESET}")
    logger.info(f"{'Weight for interior loss: ':<50}{Color.GREEN}{WEIGHT_INTERIOR}{Color.RESET}")
    logger.info(f"{'Weight for initial condition loss: ':<50}{Color.GREEN}{WEIGHT_INITIAL}{Color.RESET}")
    logger.info(f"{'Weight for boundary loss: ':<50}{Color.GREEN}{WEIGHT_BOUNDARY}{Color.RESET}")
    logger.info(f"{'Layers: ':<50}{Color.GREEN}{LAYERS}{Color.RESET}")
    logger.info(f"{'Neurons per layer: ':<50}{Color.GREEN}{NEURONS_PER_LAYER}{Color.RESET}")
    logger.info(f"{'Epochs: ':<50}{Color.GREEN}{EPOCHS}{Color.RESET}")
    logger.info(f"{'Learning rate: ':<50}{Color.GREEN}{LEARNING_RATE}{Color.RESET}")
    logger.info("="*80)
    logger.info("")

    logger.info(f"Creating B_Spline with {N_POINTS_X} points and degree {general_parameters.spline_degree}")
    b_spline = B_Splines(torch.linspace(0, 1, int(1/general_parameters.eps_interior * 5)), general_parameters.spline_degree)
    # assert check_gradient(nn_approximator, x, t)
    # to add new loss functions, add them to the list below and add the corresponding function to the array of functions in train pinn block below
    loss_iga = partial(compute_loss, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = iga_loss, dims = 1)

    # logger.info(f"Computing initial condition loss")
    # logger.info(f"{'Initial condition loss weak:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_weak, dims = 1):.12f}{Color.RESET}")
    # logger.info(f"{'Initial condition loss strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_strong, dims = 1):.12f}{Color.RESET}")
    # logger.info(f"{'Initial condition loss weak and strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_weak_and_strong, dims = 1):.12f}{Color.RESET}")
    # logger.info(f"{'Initial condition loss colocation:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_colocation, dims = 1):.12f}{Color.RESET}")

    # train the PINN
    for loss_fn, name in \
        [
            (loss_iga, 'loss_iga'), 
        ]:
        logger.info(f"Training PINN for {Color.YELLOW}{EPOCHS}{Color.RESET} epochs using {Color.YELLOW}{name}{Color.RESET} loss function")

        pinn_trained, loss_values = train_model(
            pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)
        
        if SAVE:
            SAVE_PATH = f"models/{TIMESTAMP}/{name}.pt"
            if not os.path.exists(f"models/{TIMESTAMP}"):
                os.makedirs(f"models/{TIMESTAMP}")
            logger.info(f"Saving model to {Color.YELLOW}{SAVE_PATH}{Color.RESET}")
            torch.save(pinn_trained.state_dict(), SAVE_PATH)

        compute_losses_and_plot_solution(pinn_trained=pinn_trained, x=x, device = device, \
                                        loss_values=loss_values, x_init=x_init, u_init=u_init, \
                                        N_POINTS_X=N_POINTS_X, N_POINTS_T=N_POINTS_T, \
                                        loss_fn_name=name, dims=1)


