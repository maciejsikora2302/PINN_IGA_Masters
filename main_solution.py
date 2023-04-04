import os
import torch
import argparse
import torch.nn as nn
from functools import partial
from time import time, time_ns

from PINN import PINN
from B_Splines import B_Splines
from general_parameters import general_parameters, logger, Color, TIMESTAMP, OUT_DATA_FOLDER
from utils import compute_losses_and_plot_solution
from additional_utils import get_unequaly_distribution_points
from loss_functions import interior_loss_colocation, interior_loss_strong, interior_loss_weak, interior_loss_weak_and_strong, compute_loss, initial_condition, interior_loss_weak_spline, interior_loss_weak_and_strong_spline, interior_loss_colocation_spline, interior_loss_strong_spline, compute_loss_spline, loss_PINN_learns_coeff, boundary_loss_PINN_learns_coeff, compute_loss_PINN_learns_coeff
from NN_tools import train_model
from pprint import pprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=float)
parser.add_argument("--total_time", type=float)
parser.add_argument("--n_points_x", type=int)
parser.add_argument("--n_points_t", type=int)
parser.add_argument("--n_points_init", type=int)
parser.add_argument("--weight_interior", '--wi', type=float)
parser.add_argument("--weight_initial", '--winit', type=float)
parser.add_argument("--weight_boundary", '--wb', type=float)
parser.add_argument("--layers", '-l', type=int)
parser.add_argument("--neurons_per_layer", '--npl', type=int)
parser.add_argument("--epochs", '-e', type=int)
parser.add_argument("--learning_rate", '--lr', type=float)
parser.add_argument("--eps_interior", '--eps_inter', type=float)
parser.add_argument("--spline_degree", type=int)
parser.add_argument("--save", '-s', action="store_true")
parser.add_argument("--one_dimension", '-o', action="store_true")
parser.add_argument("--uneven_distribution", '-u', action="store_true")
parser.add_argument("--optimize_test_function", '-otf', action="store_true")
training_mode = parser.add_mutually_exclusive_group(required=True)
training_mode.add_argument("--splines", action="store_true")
training_mode.add_argument("--pinn_is_solution", action="store_true")
training_mode.add_argument("--pinn_learns_coeff", action="store_true")
args = parser.parse_args()



# pprint(vars(args))



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
general_parameters.one_dimension = args.one_dimension if args.one_dimension is not None else general_parameters.one_dimension
general_parameters.uneven_distribution = args.uneven_distribution if args.uneven_distribution is not None else general_parameters.uneven_distribution
general_parameters.splines = args.splines if args.splines is not None else general_parameters.splines
general_parameters.pinn_is_solution = args.pinn_is_solution if args.pinn_is_solution is not None else general_parameters.pinn_is_solution
general_parameters.pinn_learns_coeff = args.pinn_learns_coeff if args.pinn_learns_coeff is not None else general_parameters.pinn_learns_coeff
general_parameters.optimize_test_function = args.optimize_test_function if args.optimize_test_function is not None else general_parameters.optimize_test_function

general_parameters.precalculate()
# pprint(vars(general_parameters))

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
KNOT_VECTOR = general_parameters.knot_vector
SPLINE_DEGREE = general_parameters.spline_degree
SAVE = general_parameters.save
USE_SPLINE = general_parameters.splines
OPTIMIZE_TEST_FUNCTION = general_parameters.optimize_test_function
PINN_LEARNS_coeff = general_parameters.pinn_learns_coeff
N_SPLINE_coeff = general_parameters.n_coeff


if __name__ == "__main__":


    x_domain = [0.0, LENGTH]


    if general_parameters.uneven_distribution:
        # ATTENTION, UNEVEN DISTRIBUTION WILL NOT WORK WHEN PINN IS A SOLUTION FLAG IS PROVIDED
        # SINCE I MADE ASSUMPTION THAT WHEN WE ARE LEARNING SOLUTION WE ARE USING RANDOM POINTS 
        # FROM ALL OVER THE RANGE THIS IS STILL SUBJECT TO CHANGE
        x_raw = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = N_POINTS_X, density_range=0.2, device=device)
        x_raw = x_raw.requires_grad_(True)
    else:
        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)

    if general_parameters.one_dimension:
        logger.info(f"{Color.GREEN}One dimentional problem{Color.RESET}")

        x = x_raw.flatten().reshape(-1, 1).to(device)


    else:
        logger.info(f"{Color.GREEN}Two dimentional problem{Color.RESET}")

        t_domain = [0.0, TOTAL_TIME]

        # x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=N_POINTS_T, requires_grad=True, device=device)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x = grids[0].flatten().reshape(-1, 1).to(device)
        t = grids[1].flatten().reshape(-1, 1).to(device)


    if general_parameters.uneven_distribution:
        x_init = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = N_POINTS_X, density_range=0.2, device=device)
    else:
        x_init = torch.linspace(0.0, 1.0, steps=N_POINTS_X)
    # x_init = 0.5*((x_init-0.5*LENGTH)*2)**3 + 0.5
    # x_init = x_init*LENGTH
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
    if not general_parameters.one_dimension: logger.info(f"{'Number of points in t: ':<50}{Color.GREEN}{N_POINTS_T}{Color.RESET}")
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


    if OPTIMIZE_TEST_FUNCTION:
                TEST_FUNCTION = B_Splines(
                    KNOT_VECTOR,
                    SPLINE_DEGREE,
                    dims=1 if general_parameters.one_dimension else 2
                )
    else:
        TEST_FUNCTION = None

    if USE_SPLINE:

        logger.info(f"Creating {Color.GREEN}{'1D' if general_parameters.one_dimension else '2D'}{Color.RESET} BSpline")
        spline = B_Splines(KNOT_VECTOR, degree=SPLINE_DEGREE, dims=1 if general_parameters.one_dimension else 2)
    
    elif PINN_LEARNS_coeff:
        logger.info(f"Creating PINN to learn spline coefficients with {Color.GREEN}{LAYERS}{Color.RESET} layers and {Color.GREEN}{NEURONS_PER_LAYER}{Color.RESET} neurons per layer")
        
        if general_parameters.one_dimension:
            pinn = PINN(
                LAYERS, 
                NEURONS_PER_LAYER, 
                pinning=False, 
                act=nn.Tanh(), 
                dim_layer_in=1, 
                dim_layer_out=N_SPLINE_coeff
                ).to(device)

            # In this case the coefficients don't matter
            spline = B_Splines(KNOT_VECTOR, degree=SPLINE_DEGREE, dims=1)
        elif not general_parameters.one_dimension:
            pinn = PINN(
                LAYERS, 
                NEURONS_PER_LAYER, 
                pinning=False, 
                act=nn.Tanh(), 
                dim_layer_in=x.shape[0], # Dim layer in 2D case needs to be modified in future 
                dim_layer_out=N_SPLINE_coeff,
                pinn_learns_coeff=general_parameters.pinn_learns_coeff,
                ).to(device)
            
            spline = B_Splines(KNOT_VECTOR, degree=SPLINE_DEGREE, dims=2)

        else:
            raise Exception("Unknown dimensionality")
    else:
        
        logger.info(f"Creating PINN with {Color.GREEN}{LAYERS}{Color.RESET} layers and {Color.GREEN}{NEURONS_PER_LAYER}{Color.RESET} neurons per layer")

        if general_parameters.one_dimension:
            pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=False, act=nn.Tanh(), pinn_learns_coeff=general_parameters.pinn_learns_coeff).to(device)
        elif not general_parameters.one_dimension:
            pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=False, act=nn.Tanh(), pinn_learns_coeff=general_parameters.pinn_learns_coeff, dims=2).to(device)
        else:
            raise Exception("Unknown dimensionality")

    # assert check_gradient(nn_approximator, x, t)
    # to add new loss functions, add them to the list below and add the corresponding function to the array of functions in train pinn block below
    
    if general_parameters.pinn_is_solution or general_parameters.splines:

        loss_fn_weak = partial(
            compute_loss if not USE_SPLINE else compute_loss_spline, 
            x=x, 
            weight_f=WEIGHT_INTERIOR, 
            weight_i=WEIGHT_INITIAL, 
            weight_b=WEIGHT_BOUNDARY, 
            interior_loss_function=interior_loss_weak if not USE_SPLINE else interior_loss_weak_spline, 
            dims=1 if general_parameters.one_dimension else 2,
            test_function=TEST_FUNCTION
        )

        loss_fn_strong = partial(
            compute_loss if not USE_SPLINE else compute_loss_spline, 
            x=x, 
            weight_f=WEIGHT_INTERIOR, 
            weight_i=WEIGHT_INITIAL, 
            weight_b=WEIGHT_BOUNDARY, 
            interior_loss_function=interior_loss_strong if not USE_SPLINE else interior_loss_strong_spline, 
            dims=1 if general_parameters.one_dimension else 2,
            test_function=TEST_FUNCTION
        )

        loss_fn_weak_and_strong = partial(
            compute_loss if not USE_SPLINE else compute_loss_spline, 
            x=x, 
            weight_f=WEIGHT_INTERIOR, 
            weight_i=WEIGHT_INITIAL, 
            weight_b=WEIGHT_BOUNDARY, 
            interior_loss_function=interior_loss_weak_and_strong if not USE_SPLINE else interior_loss_weak_and_strong_spline, 
            dims=1 if general_parameters.one_dimension else 2,
            test_function=TEST_FUNCTION
        )

        loss_fn_colocation = partial(
            compute_loss if not USE_SPLINE else compute_loss_spline, 
            x=x, 
            weight_f=WEIGHT_INTERIOR, 
            weight_i=WEIGHT_INITIAL, 
            weight_b=WEIGHT_BOUNDARY, 
            interior_loss_function=interior_loss_colocation if not USE_SPLINE else interior_loss_colocation_spline, 
            dims=1 if general_parameters.one_dimension else 2
        )


        # logger.info(f"Computing initial condition loss")
        # logger.info(f"{'Initial condition loss weak:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_weak, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_strong, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss weak and strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_weak_and_strong, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss colocation:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INTERIOR, weight_b=WEIGHT_BOUNDARY, interior_loss_function = interior_loss_colocation, dims = 1):.12f}{Color.RESET}")

        # train the PINN
        for loss_fn, name in \
            [
                (loss_fn_weak, 'loss_fn_weak'),
                # (loss_fn_strong, 'loss_fn_strong'), 
                # (loss_fn_weak_and_strong, 'loss_fn_weak_and_strong'), 
                # (loss_fn_colocation, 'loss_fn_colocation')
            ]:
            logger.info(f"Training {'PINN' if not USE_SPLINE else 'splines'} for {Color.YELLOW}{EPOCHS}{Color.RESET} epochs using {Color.YELLOW}{name}{Color.RESET} loss function")

            model = pinn if not USE_SPLINE else spline

            start_time = time()
            model_trained, loss_values = train_model(
                model,
                loss_fn=loss_fn, 
                loss_fn_name=name, 
                learning_rate=LEARNING_RATE, 
                max_epochs=EPOCHS,
                test_function=TEST_FUNCTION)
            end_time = time()

            training_time = end_time - start_time

            logger.info(f"Training took {Color.GREEN}{training_time:.2f}{Color.RESET} seconds")

            compute_losses_and_plot_solution(
                pinn_trained=model_trained,\
                x=x,\
                device = device, \
                loss_values=loss_values, \
                x_init=x_init, \
                u_init=u_init, \
                N_POINTS_X=N_POINTS_X, \
                N_POINTS_T=N_POINTS_T, \
                loss_fn_name=name, \
                training_time=training_time, \
                dims=1 if general_parameters.one_dimension else 2
            )
    elif general_parameters.pinn_learns_coeff:

        loss_fn = partial(
            compute_loss_PINN_learns_coeff,
            x=x,
            spline=spline,
            weight_f=WEIGHT_INTERIOR, 
            weight_b=WEIGHT_BOUNDARY, 
            dims=1 if general_parameters.one_dimension else 2
        )
           
        logger.info(f"Training PINN to coefficients estimation for {Color.YELLOW}{EPOCHS}{Color.RESET} epochs using {Color.YELLOW}{loss_fn}{Color.RESET} loss function")

        model = pinn
        name = "Prediction of splines coefficients using PINN"

        start_time = time()
        model_trained, loss_values = train_model(
            model,
            loss_fn=loss_fn, 
            loss_fn_name=name, 
            learning_rate=LEARNING_RATE, 
            max_epochs=EPOCHS,
            test_function=TEST_FUNCTION)
        end_time = time()

        training_time = end_time - start_time

        logger.info(f"Training took {Color.GREEN}{training_time:.2f}{Color.RESET} seconds")

        compute_losses_and_plot_solution(
            pinn_trained=model_trained,\
            x=x,\
            device = device, \
            loss_values=loss_values, \
            x_init=x_init, \
            u_init=u_init, \
            N_POINTS_X=N_POINTS_X, \
            N_POINTS_T=N_POINTS_T, \
            loss_fn_name=name, \
            training_time=training_time, \
            dims=1 if general_parameters.one_dimension else 2
        )