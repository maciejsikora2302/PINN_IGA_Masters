import torch
import os
import numpy as np
from general_parameters import logger, Color, OUT_DATA_FOLDER, general_parameters
import matplotlib.pyplot as plt
from NN_tools import f
from PINN import PINN
from scipy.interpolate import BSpline
import json
from matplotlib import cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def compute_losses_and_plot_solution(
        pinn_trained: PINN,
        x: torch.Tensor,
        device: torch.device, 
        loss_values: np.ndarray, 
        x_init: torch.Tensor, 
        N_POINTS_X: int, 
        N_POINTS_T: int, 
        loss_fn_name: str, 
        training_time: float,
        t: torch.Tensor = None, 
        dims: int = 2
    ):


    x = torch.linspace(0, 1, N_POINTS_X).reshape(-1, 1).to(device)
    t = torch.linspace(0, 1, N_POINTS_T).reshape(-1, 1).to(device) if t is not None else t

    # logger.info(f"{Color.RED}x: {x}{Color.RESET}")
    # logger.info(f"{Color.RED}t: {t}{Color.RESET}")

    if pinn_trained is None or x is None or device is None or loss_values is None or x_init is None or N_POINTS_X is None or N_POINTS_T is None or loss_fn_name is None:
        logger.info(f"{Color.RED}One of the parameters is None{Color.RESET}")
        logger.info(f"{Color.RED}pinn_trained: {pinn_trained}{Color.RESET}")
        logger.info(f"{Color.RED}x: {x}{Color.RESET}")
        logger.info(f"{Color.RED}t: {t}{Color.RESET}")
        logger.info(f"{Color.RED}device: {device}{Color.RESET}")
        logger.info(f"{Color.RED}loss_values: {loss_values}{Color.RESET}")
        logger.info(f"{Color.RED}x_init: {x_init}{Color.RESET}")
        logger.info(f"{Color.RED}N_POINTS_X: {N_POINTS_X}{Color.RESET}")
        logger.info(f"{Color.RED}N_POINTS_T: {N_POINTS_T}{Color.RESET}")
        logger.info(f"{Color.RED}loss_fn_name: {loss_fn_name}{Color.RESET}")
        raise ValueError("One of the parameters is None, for 1D case t can be None")

    path = f"{OUT_DATA_FOLDER}/{loss_fn_name}"

    if not os.path.exists(path):
        os.makedirs(path)


    logger.info(f"Creating plots and saving to files. Dimensions: {dims}")

    #save time to file
    with open(f"{path}/time.txt", "w") as time_file:
        time_file.write(f"{training_time:.2f}")

    #if loss values is vertical vector, make it horizontal
    if loss_values is not None and len(loss_values.shape) == 3:
        loss_values = loss_values.flatten()

    #save loss values to file
    with open(f"{path}/loss_values.txt", "w") as loss_file:
        for loss in loss_values:
            loss_file.write(f"{loss:.5f},")
    


                            


    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Loss function convergence")
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Loss function value in log scale")
    #plot loss in log10 scale
    ax.plot(np.log10(loss_values))

    plt.savefig(f"{path}/loss_convergence.png")



    if dims == 1:
        if general_parameters.pinn_is_solution:
            pinn_values = f(pinn_trained.to(device), x.reshape(-1, 1)).cpu().flatten().detach()
        elif general_parameters.pinn_learns_coeff:
            
            coeff = []

            for eps in general_parameters.epsilon_list:
                eps = torch.Tensor(eps).unsqueeze(0)
                for pinn in pinn_trained:
                    coeff.append(
                        f(pinn, eps).cpu().flatten().detach()
                    )

            spline = BSpline(general_parameters.knot_vector, coeff, general_parameters.spline_degree)

            # It's a spline, which coefficients were predicted by PINN
            pinn_values = torch.Tensor(spline(x.cpu().detach())).flatten()
    else:

        pinn_values = f(pinn_trained.to(device), x.reshape(-1, 1), t.reshape(-1,1))
        pinn_values = pinn_values.reshape(N_POINTS_X, N_POINTS_T)



    #save x to file
    with open(f"{path}/x.txt", "w") as x_file:
        for x_value in x_init:
            x_file.write(f"{x_value:.5f},")

    #save pinn values to file
    torch.save(pinn_values, f"{path}/pinn_values.pt")
    try:
        with open(f"{path}/pinn_values.txt", "w") as pinn_values_file:
            for pinn_value in pinn_values:
                pinn_values_file.write(f"{pinn_value:.5f},")
    except:
        pass

    with open(f"{path}/model_parameters.txt", "w") as file:
        file.write(str(pinn_trained))

    with open(f"{path}/other_parameters.txt", "w") as file:
        dict_to_save = {
            "pinn_is_solution": general_parameters.pinn_is_solution,
            "pinn_learns_coeff": general_parameters.pinn_learns_coeff,
            "epochs": general_parameters.epochs,
            "learning_rate": general_parameters.learning_rate,
            "eps_interior": general_parameters.eps_interior,
            "n_points_x": general_parameters.n_points_x,
            "weight_initial": general_parameters.weight_initial,
            "weight_boundary": general_parameters.weight_boundary,
            "weight_interior": general_parameters.weight_interior,
            "layers": general_parameters.layers,
            "neurons": general_parameters.neurons_per_layer,
            "spline_degree": general_parameters.spline_degree,
        }
        file.write(json.dumps(dict_to_save))

    if dims == 1:

        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
        ax.set_title("Solution profile")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.plot(x_init.cpu(), pinn_values, label="PINN solution")
        # if general_parameters.pinn_learns_coeff:
        #     ax.plot(x_init.cpu(), f(pinn_trained.to(device), x).cpu().flatten().detach(), label="Learned coeffs")
        ax.legend()
        plt.savefig(f"{path}/solution_profile.png")


        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
        ax.set_title("Solution profile (normalized)")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_ylim(-0.1, 1.1)
        ax.plot(x_init.cpu(), pinn_values, label="PINN solution")
        # if general_parameters.pinn_learns_coeff:
        #     ax.plot(x_init.cpu(), f(pinn_trained.to(device), x).cpu().flatten().detach(), label="Learned coeffs")
        ax.legend()
        plt.savefig(f"{path}/solution_profile_normalized.png")
        
        
        with open(f"{path}/solution_profile.txt", "w") as file:
            file.write(','.join([str(x) for x in x_init]))
            file.write('\n')
            y = pinn_values
            file.write(','.join([str(x) for x in y]))


        #write parameters of pinn clas to file
        
        # write other important parameters to file

    if dims == 2:
        X, T = np.meshgrid(x.cpu(), t.cpu())

        fig = plt.figure(figsize=(14, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Solution profile")
        ax.plot_surface(X, T, pinn_values.cpu().detach().numpy(), cmap=cm.coolwarm , linewidth=0, antialiased=False)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u")
        plt.savefig(f"{path}/solution_profile.png")



        fig = plt.figure(figsize=(14, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Solution profile (normalized)")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u")
        ax.set_zlim(-0.1, 1.1)
        ax.plot_surface(X, T, pinn_values.cpu().detach().numpy(), cmap=cm.coolwarm , linewidth=0, antialiased=False)
        plt.savefig(f"{path}/solution_profile_normalized.png")
