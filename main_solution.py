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
from loss_functions import interior_loss_basic ,interior_loss_strong, interior_loss_weak, interior_loss_weak_and_strong, compute_loss, initial_condition
from NN_tools import train_model
from pprint import pprint


def train_and_plot(model, loss_fn, loss_fn_name, x, x_init, t, test_function):
    logger.info(f"Training model for {Color.YELLOW}{general_parameters.epochs}{Color.RESET} epochs using {Color.YELLOW}{loss_fn_name}{Color.RESET} loss function")

    start_time = time()
    model_trained, loss_values = train_model(
        model,
        loss_fn=loss_fn,
        loss_fn_name=loss_fn_name,
        learning_rate=general_parameters.learning_rate,
        max_epochs=general_parameters.epochs,
        test_function=test_function)
    end_time = time()

    training_time = end_time - start_time

    logger.info(f"Training took {Color.GREEN}{training_time:.2f}{Color.RESET} seconds")

    compute_losses_and_plot_solution(
        pinn_trained=model_trained,
        x=x,
        t=t,
        device=device,
        loss_values=loss_values,
        x_init=x_init,
        N_POINTS_X=general_parameters.n_points_x,
        N_POINTS_T=general_parameters.n_points_t,
        loss_fn_name=loss_fn_name,
        training_time=training_time,
        dims=1 if general_parameters.one_dimension else 2,
        test_function=test_function
    )

def get_model():

    if general_parameters.splines:
        logger.info(f"Creating {Color.GREEN}{'1D' if general_parameters.one_dimension else '2D'}{Color.RESET} BSpline")
        spline = B_Splines(general_parameters.knot_vector, degree=general_parameters.spline_degree, dims=1 if general_parameters.one_dimension else 2).to(device)
        return spline
    
    elif general_parameters.pinn_is_solution:
        logger.info(f"Creating PINN with {Color.GREEN}{general_parameters.layers}{Color.RESET} layers and {Color.GREEN}{general_parameters.neurons_per_layer}{Color.RESET} neurons per layer")
        pinn = PINN(
            general_parameters.layers, 
            general_parameters.neurons_per_layer, 
            pinning=False, 
            act=nn.Tanh(), 
            pinn_learns_coeff=general_parameters.pinn_learns_coeff, 
            dims=1 if general_parameters.one_dimension else 2
            ).to(device)
        
        return pinn
        
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
        else:
            raise Exception("Double check this part of the code")
            pinn = PINN(
                general_parameters.layers, 
                general_parameters.neurons_per_layer, 
                pinning=False, 
                act=nn.Tanh(), 
                dim_layer_in=x.shape[0], # Dim layer in 2D case needs to be modified in future 
                dim_layer_out=general_parameters.n_coeff,
                pinn_learns_coeff=general_parameters.pinn_learns_coeff,
                ).to(device)
    
        return pinn_list
    
    else:
        raise Exception("No model has been chosen")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
logger.info(f"Device: {device}")

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=float, default=1.)
parser.add_argument("--total_time", type=float, default=1.)
parser.add_argument("--n_points_init", type=int, default=300)
parser.add_argument("--n_points_x", type=int, default=100)
parser.add_argument("--n_points_t", type=int, default=150)
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
parser.add_argument("--epsilon_list", default=torch.linspace(0.01, 0.1, 10, requires_grad=True).unsqueeze(0).mT.to(device))
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

        x_raw = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = general_parameters.n_points_x, density_range=0.2, device=device)
        x_raw = x_raw.requires_grad_(True)

        if not general_parameters.one_dimension:
            t_raw = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = general_parameters.n_points_t, density_range=0.2, device=device)
            t_raw = t_raw.requires_grad_(True)
    else:
        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=general_parameters.n_points_x, requires_grad=True)

    if general_parameters.one_dimension:
        logger.info(f"{Color.GREEN}One dimentional problem{Color.RESET}")

        x = x_raw.flatten().reshape(-1, 1).to(device)

    else:
        logger.info(f"{Color.GREEN}Two dimentional problem{Color.RESET}")

        t_domain = [0.0, 1.0]

        # x_raw = torch.linspace(x_domain[0], x_domain[1], steps=general_parameters.n_points_x, requires_grad=True)

        if not general_parameters.uneven_distribution:
            t_raw = torch.linspace(t_domain[0], t_domain[1], steps=general_parameters.n_points_t, requires_grad=True, device=device)
        # grids = torch.meshgrid(x_raw.to(device), t_raw.to(device), indexing="ij")
        
        # x = grids[0].flatten().reshape(-1, 1).to(device)
        # t = grids[1].flatten().reshape(-1, 1).to(device)

        x = x_raw.flatten().reshape(-1, 1).to(device)
        t = t_raw.flatten().reshape(-1, 1).to(device)

    if general_parameters.uneven_distribution:
        x_init = get_unequaly_distribution_points(eps=general_parameters.eps_interior, n = general_parameters.n_points_x, density_range=0.2, device=device)
    else:
        x_init = torch.linspace(0.0, 1.0, steps=general_parameters.n_points_x)

    u_init = initial_condition(x_init)


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


    def get_loss_fn(loss_type, x, test_function):

        loss_fn_dict = {
            'basic': interior_loss_basic,
            'weak': interior_loss_weak,
            'strong': interior_loss_strong,
            'weak_and_strong': interior_loss_weak_and_strong
        }

        if loss_type not in loss_fn_dict.keys():
            raise Exception(f"Loss function {loss_type} not implemented")
        

        return partial(
            compute_loss,
            x=x,
            t=t if not general_parameters.one_dimension else None,
            weight_f=general_parameters.weight_interior,
            weight_i=general_parameters.weight_initial,
            weight_b=general_parameters.weight_boundary,
            interior_loss_function=loss_fn_dict[loss_type],
            dims=1 if general_parameters.one_dimension else 2,
            test_function=test_function
        )
        
    model = get_model()

    

    if general_parameters.optimize_test_function:
        test_function = B_Splines(general_parameters.knot_vector, degree=general_parameters.spline_degree, dims=1 if general_parameters.one_dimension else 2)
    else:
        test_function = None

    loss_fn_basic = get_loss_fn('basic', x, test_function=None)
    loss_fn_weak = get_loss_fn('weak', x, test_function)
    loss_fn_strong = get_loss_fn('strong', x, test_function)
    loss_fn_weak_and_strong = get_loss_fn('weak_and_strong', x, test_function)
    


    loss_functions = [
        (loss_fn_basic, 'loss_fn_basic'),
        (loss_fn_weak, 'loss_fn_weak'),
        (loss_fn_strong, 'loss_fn_strong'),
        (loss_fn_weak_and_strong, 'loss_fn_weak_and_strong'),
    ]

    for loss_fn, name in loss_functions:
        train_and_plot(model, loss_fn, name, x, x_init, t if not general_parameters.one_dimension else None, test_function)