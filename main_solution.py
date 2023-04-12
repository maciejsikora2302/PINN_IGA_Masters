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
parser.add_argument("--length", type=float, default=1.)
parser.add_argument("--total_time", type=float, default=1.)
parser.add_argument("--n_points_x", type=int, default=100)
parser.add_argument("--n_points_t", type=int, default=150)
parser.add_argument("--n_points_init", type=int, default=300)
parser.add_argument("--weight_interior", '--wi', type=float, default=50.0)
parser.add_argument("--weight_initial", '--winit', type=float, default=.5)
parser.add_argument("--weight_boundary", '--wb', type=float, default=5.0)
parser.add_argument("--layers", '-l', type=int, default=4)
parser.add_argument("--neurons_per_layer", '--npl', type=int, default=20)
parser.add_argument("--epochs", '-e', type=int, default=50_000)
parser.add_argument("--learning_rate", '--lr', type=float, default=0.0025)
parser.add_argument("--eps_interior", '--eps_inter', type=float, default=1e-1)
parser.add_argument("--spline_degree", type=int, default=3)
parser.add_argument("--save", '-s', action="store_true", default=False)
parser.add_argument("--one_dimension", '-o', action="store_true", default=False)
parser.add_argument("--uneven_distribution", '-u', action="store_true", default=False)
parser.add_argument("--optimize_test_function", '-otf', action="store_true", default=False)
parser.add_argument("--epsilon_list", default=torch.linspace(0.01, 0.1, 10, requires_grad=True).unsqueeze(0).mT.cuda())
parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

training_mode = parser.add_mutually_exclusive_group()
training_mode.add_argument("--splines", action="store_true", default=False)
training_mode.add_argument("--pinn_is_solution", action="store_true", default=False)
training_mode.add_argument("--pinn_learns_coeff", action="store_true", default=False)
args = parser.parse_args()

general_parameters.__init__(args)
general_parameters.precalculate()




if __name__ == "__main__":


    x_domain = [0.0, general_parameters.length]


    if general_parameters.uneven_distribution:
        # ATTENTION, UNEVEN DISTRIBUTION WILL NOT WORK WHEN PINN IS A SOLUTION FLAG IS PROVIDED
        # SINCE I MADE ASSUMPTION THAT WHEN WE ARE LEARNING SOLUTION WE ARE USING RANDOM POINTS 
        # FROM ALL OVER THE RANGE THIS IS STILL SUBJECT TO CHANGE
        x_raw = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = general_parameters.n_points_x, density_range=0.2, device=device)
        x_raw = x_raw.requires_grad_(True)
    else:
        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=general_parameters.n_points_x, requires_grad=True)

    if general_parameters.one_dimension:
        logger.info(f"{Color.GREEN}One dimentional problem{Color.RESET}")

        x = x_raw.flatten().reshape(-1, 1).to(device)


    else:
        logger.info(f"{Color.GREEN}Two dimentional problem{Color.RESET}")

        t_domain = [0.0, general_parameters.total_time]

        # x_raw = torch.linspace(x_domain[0], x_domain[1], steps=general_parameters.n_points_x, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=general_parameters.n_points_t, requires_grad=True, device=device)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x = grids[0].flatten().reshape(-1, 1).to(device)
        t = grids[1].flatten().reshape(-1, 1).to(device)


    if general_parameters.uneven_distribution:
        x_init = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = general_parameters.n_points_x, density_range=0.2, device=device)
    else:
        x_init = torch.linspace(0.0, 1.0, steps=general_parameters.n_points_x)
    # x_init = 0.5*((x_init-0.5*general_parameters.length)*2)**3 + 0.5
    # x_init = x_init*general_parameters.length
    u_init = initial_condition(x_init)

    # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    # ax.set_title("Initial condition points")
    # ax.set_xlabel("x")
    # ax.set_ylabel("u")
    # ax.scatter(x_init, u_init, s=2)

    logger.info("")
    logger.info("="*80)
    logger.info("Learning parameters")
    logger.info(f"{'Length: ':<50}{Color.GREEN}{general_parameters.length}{Color.RESET}")
    logger.info(f"{'Total time: ':<50}{Color.GREEN}{general_parameters.total_time}{Color.RESET}")
    logger.info(f"{'Number of points in x: ':<50}{Color.GREEN}{general_parameters.n_points_x}{Color.RESET}")
    if not general_parameters.one_dimension: logger.info(f"{'Number of points in t: ':<50}{Color.GREEN}{general_parameters.n_points_t}{Color.RESET}")
    logger.info(f"{'Number of points in initial condition: ':<50}{Color.GREEN}{general_parameters.n_points_init}{Color.RESET}")
    logger.info(f"{'Weight for interior loss: ':<50}{Color.GREEN}{general_parameters.weight_interior}{Color.RESET}")
    logger.info(f"{'Weight for initial condition loss: ':<50}{Color.GREEN}{general_parameters.weight_initial}{Color.RESET}")
    logger.info(f"{'Weight for boundary loss: ':<50}{Color.GREEN}{general_parameters.weight_boundary}{Color.RESET}")
    logger.info(f"{'Layers: ':<50}{Color.GREEN}{general_parameters.layers}{Color.RESET}")
    logger.info(f"{'Neurons per layer: ':<50}{Color.GREEN}{general_parameters.neurons_per_layer}{Color.RESET}")
    logger.info(f"{'Epochs: ':<50}{Color.GREEN}{general_parameters.epochs}{Color.RESET}")
    logger.info(f"{'Learning rate: ':<50}{Color.GREEN}{general_parameters.learning_rate}{Color.RESET}")
    logger.info("="*80)
    logger.info("")


    if general_parameters.optimize_test_function:
                TEST_FUNCTION = B_Splines(
                    general_parameters.knot_vector,
                    general_parameters.spline_degree,
                    dims=1 if general_parameters.one_dimension else 2
                )
    else:
        TEST_FUNCTION = None

    if general_parameters.splines:

        logger.info(f"Creating {Color.GREEN}{'1D' if general_parameters.one_dimension else '2D'}{Color.RESET} BSpline")
        spline = B_Splines(general_parameters.knot_vector, degree=general_parameters.spline_degree, dims=1 if general_parameters.one_dimension else 2)
    
    elif general_parameters.pinn_learns_coeff:
        logger.info(f"Creating PINN to learn spline coefficients with {Color.GREEN}{general_parameters.layers}{Color.RESET} layers and {Color.GREEN}{general_parameters.neurons_per_layer}{Color.RESET} neurons per layer")
        
        if general_parameters.one_dimension:
            
            pinn_list = [
                PINN(
                    general_parameters.layers, 
                    general_parameters.neurons_per_layer, 
                    pinning=False, 
                    act=nn.Tanh(), 
                    dim_layer_in=1, 
                    dim_layer_out=1
                ).to(device) for _ in range(general_parameters.n_coeff)
            ]


            # In this case the coefficients don't matter
            spline = B_Splines(general_parameters.knot_vector, degree=general_parameters.spline_degree, dims=1)
        elif not general_parameters.one_dimension:
            pinn = PINN(
                general_parameters.layers, 
                general_parameters.neurons_per_layer, 
                pinning=False, 
                act=nn.Tanh(), 
                dim_layer_in=x.shape[0], # Dim layer in 2D case needs to be modified in future 
                dim_layer_out=general_parameters.n_coeff,
                pinn_learns_coeff=general_parameters.pinn_learns_coeff,
                ).to(device)
            
            spline = B_Splines(general_parameters.knot_vector, degree=general_parameters.spline_degree, dims=2)

        else:
            raise Exception("Unknown dimensionality")
    else:
        
        logger.info(f"Creating PINN with {Color.GREEN}{general_parameters.layers}{Color.RESET} layers and {Color.GREEN}{general_parameters.neurons_per_layer}{Color.RESET} neurons per layer")

        if general_parameters.one_dimension:
            pinn = PINN(general_parameters.layers, general_parameters.neurons_per_layer, pinning=False, act=nn.Tanh(), pinn_learns_coeff=general_parameters.pinn_learns_coeff).to(device)
        elif not general_parameters.one_dimension:
            pinn = PINN(general_parameters.layers, general_parameters.neurons_per_layer, pinning=False, act=nn.Tanh(), pinn_learns_coeff=general_parameters.pinn_learns_coeff, dims=2).to(device)
        else:
            raise Exception("Unknown dimensionality")

    # assert check_gradient(nn_approximator, x, t)
    # to add new loss functions, add them to the list below and add the corresponding function to the array of functions in train pinn block below
    
    if general_parameters.pinn_is_solution or general_parameters.splines:

        def get_loss_fn(loss_type, general_parameters, x, test_function):
            if loss_type == 'weak':
                interior_loss_func = interior_loss_weak_spline if general_parameters.splines else interior_loss_weak
            elif loss_type == 'strong':
                interior_loss_func = interior_loss_strong_spline if general_parameters.splines else interior_loss_strong
            elif loss_type == 'weak_and_strong':
                interior_loss_func = interior_loss_weak_and_strong_spline if general_parameters.splines else interior_loss_weak_and_strong
            elif loss_type == 'colocation':
                interior_loss_func = interior_loss_colocation_spline if general_parameters.splines else interior_loss_colocation
            else:
                raise ValueError("Invalid loss_type provided.")
            
            compute_loss_func = compute_loss_spline if general_parameters.splines else compute_loss

            return partial(
                compute_loss_func,
                x=x,
                weight_f=general_parameters.weight_interior,
                weight_i=general_parameters.weight_initial,
                weight_b=general_parameters.weight_boundary,
                interior_loss_function=interior_loss_func,
                dims=1 if general_parameters.one_dimension else 2,
                test_function=test_function
            )

        loss_fn_weak = get_loss_fn('weak', general_parameters, x, TEST_FUNCTION)
        loss_fn_strong = get_loss_fn('strong', general_parameters, x, TEST_FUNCTION)
        loss_fn_weak_and_strong = get_loss_fn('weak_and_strong', general_parameters, x, TEST_FUNCTION)
        loss_fn_colocation = get_loss_fn('colocation', general_parameters, x, TEST_FUNCTION)



        # logger.info(f"Computing initial condition loss")
        # logger.info(f"{'Initial condition loss weak:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=general_parameters.weight_interior, weight_i=general_parameters.weight_interior, weight_b=general_parameters.weight_boundary, interior_loss_function = interior_loss_weak, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=general_parameters.weight_interior, weight_i=general_parameters.weight_interior, weight_b=general_parameters.weight_boundary, interior_loss_function = interior_loss_strong, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss weak and strong:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=general_parameters.weight_interior, weight_i=general_parameters.weight_interior, weight_b=general_parameters.weight_boundary, interior_loss_function = interior_loss_weak_and_strong, dims = 1):.12f}{Color.RESET}")
        # logger.info(f"{'Initial condition loss colocation:':<50}{Color.GREEN}{compute_loss(pinn, x=x, weight_f=general_parameters.weight_interior, weight_i=general_parameters.weight_interior, weight_b=general_parameters.weight_boundary, interior_loss_function = interior_loss_colocation, dims = 1):.12f}{Color.RESET}")

        # train the PINN
        for loss_fn, name in \
            [
                # (loss_fn_weak, 'loss_fn_weak'),
                # (loss_fn_strong, 'loss_fn_strong'), 
                (loss_fn_weak_and_strong, 'loss_fn_weak_and_strong'), 
                # (loss_fn_colocation, 'loss_fn_colocation')
            ]:
            logger.info(f"Training {'PINN' if not general_parameters.splines else 'splines'} for {Color.YELLOW}{general_parameters.epochs}{Color.RESET} epochs using {Color.YELLOW}{name}{Color.RESET} loss function")

            model = pinn if not general_parameters.splines else spline

            start_time = time()
            model_trained, loss_values = train_model(
                model,
                loss_fn=loss_fn, 
                loss_fn_name=name, 
                learning_rate=general_parameters.learning_rate, 
                max_epochs=general_parameters.epochs,
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
                general_parameters.n_points_x=general_parameters.n_points_x, \
                general_parameters.n_points_t=general_parameters.n_points_t, \
                loss_fn_name=name, \
                training_time=training_time, \
                dims=1 if general_parameters.one_dimension else 2
            )
    elif general_parameters.pinn_learns_coeff:

        loss_fn = partial(
            compute_loss_PINN_learns_coeff,
            x=x,
            spline=spline,
            weight_f=general_parameters.weight_interior, 
            weight_b=general_parameters.weight_boundary, 
            dims=1 if general_parameters.one_dimension else 2,
            test_function = TEST_FUNCTION
        )
           
        logger.info(f"Training PINN to coefficients estimation for {Color.YELLOW}{general_parameters.epochs}{Color.RESET} epochs using {Color.YELLOW}{loss_fn}{Color.RESET} loss function")

        model = pinn_list
        name = "Prediction of splines coefficients using PINN"

        start_time = time()
        model_trained, loss_values = train_model(
            model,
            loss_fn=loss_fn, 
            loss_fn_name=name, 
            learning_rate=general_parameters.learning_rate, 
            max_epochs=general_parameters.epochs,
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
            general_parameters.n_points_x=general_parameters.n_points_x, \
            general_parameters.n_points_t=general_parameters.n_points_t, \
            loss_fn_name=name, \
            training_time=training_time, \
            dims=1 if general_parameters.one_dimension else 2
        )