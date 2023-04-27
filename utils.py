import torch
import os
import numpy as np
from general_parameters import logger, Color, OUT_DATA_FOLDER, general_parameters
import matplotlib.pyplot as plt
from NN_tools import f
from PINN import PINN
from scipy.interpolate import BSpline
import json

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
        u_init: torch.Tensor, 
        N_POINTS_X: int, 
        N_POINTS_T: int, 
        loss_fn_name: str, 
        training_time: float,
        t: torch.Tensor = None, 
        dims: int = 2
    ):
    # print(f"type of pinn_trained is {type(pinn_trained)}")
    # print(f"type of x is {type(x)}")
    # print(f"type of device is {type(device)}")
    # print(f"type of loss_values is {type(loss_values)}")
    # print(f"type of x_init is {type(x_init)}")
    # print(f"type of u_init is {type(u_init)}")
    # print(f"type of N_POINTS_X is {type(N_POINTS_X)}")
    # print(f"type of N_POINTS_T is {type(N_POINTS_T)}")
    # print(f"type of loss_fn_name is {type(loss_fn_name)}")
    # print(f"type of t is {type(t)}")
    # check if any of parameters is None
    if pinn_trained is None or x is None or device is None or loss_values is None or x_init is None or u_init is None or N_POINTS_X is None or N_POINTS_T is None or loss_fn_name is None:
        logger.info(f"{Color.RED}One of the parameters is None{Color.RESET}")
        logger.info(f"{Color.RED}pinn_trained: {pinn_trained}{Color.RESET}")
        logger.info(f"{Color.RED}x: {x}{Color.RESET}")
        logger.info(f"{Color.RED}t: {t}{Color.RESET}")
        logger.info(f"{Color.RED}device: {device}{Color.RESET}")
        logger.info(f"{Color.RED}loss_values: {loss_values}{Color.RESET}")
        logger.info(f"{Color.RED}x_init: {x_init}{Color.RESET}")
        logger.info(f"{Color.RED}u_init: {u_init}{Color.RESET}")
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

    #save loss values to file
    with open(f"{path}/loss_values.txt", "w") as loss_file:
        for loss in loss_values:
            loss_file.write(f"{loss:.5f},")
    


                            

    # logger.info(f"Computing final loss")

    # losses = compute_loss(pinn_trained.to(device), x=x, t=t, verbose=True, dims=dims)
    # logger.info(f"{'Total loss:':<50}{Color.GREEN}{losses[0]:.5f}{Color.RESET}    ({losses[0]:.3E})")
    # logger.info(f"{'Interior loss:':<50}{Color.GREEN}{losses[1]:.5f}{Color.RESET}    ({losses[1]:.3E})")
    # logger.info(f"{'Initial loss:':<50}{Color.GREEN}{losses[2]:.5f}{Color.RESET}    ({losses[2]:.3E})")
    # logger.info(f"{'Bondary loss:':<50}{Color.GREEN}{losses[3]:.5f}{Color.RESET}    ({losses[3]:.3E})")

    # average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Loss function convergence")
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Loss function value in log scale")
    # ax.plot(average_loss)
    #plot loss in log10 scale
    ax.plot(np.log10(loss_values))

    plt.savefig(f"{path}/loss_convergence.png")

    
    # z = f(pinn_trained.to(device), x, t)
    # color = plot_color(z.cpu(), x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T)
    # plt.plot(x_init, u_init, label="Initial condition")
    # plt.plot(x_init, pinn_values.flatten().detach(), label="PINN solution")
    # plt.legend()
    # plt.savefig("./imgs/initial_condition2.png")

    # print(pinn_trained.to(device))
    # print(x_init.reshape(-1, 1))
    # print(torch.zeros_like(x_init))

    # if general_parameters.pinn_learns_coeff:
    #     x = x.reshape(x.shape[0], x.shape[1])

    if dims == 1 and not general_parameters.pinn_learns_coeff:
        pinn_values = f(pinn_trained.to(device), x)
    elif dims == 2 and not general_parameters.pinn_learns_coeff:
        pinn_values = f(pinn_trained.to(device), x.reshape(-1, 1), torch.zeros_like(x_init).reshape(-1,1).to(device))

    if not general_parameters.pinn_learns_coeff:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_title("Initial condition difference")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.plot(x_init.cpu(), u_init.cpu(), label="Initial condition")
        ax.plot(x_init.cpu(), pinn_values.cpu().flatten().detach(), label="PINN solution")
        ax.legend()
        plt.savefig(f"{path}/initial_condition.png")

    # from IPython.display import HTML
    # ani = plot_solution(pinn_trained.cpu(), x.cpu(), t.cpu())
    # HTML(ani.to_html5_video())

    # plt.plot(x_init, u_init, label="Initial condition")
    # plt.plot(x_init, pinn_values.flatten().detach(), label="PINN solution")
    # plt.legend()

    # pinn_values = f(pinn_trained.to(device), torch.zeros_like(x_init).reshape(-1,1)+0.5, x_init.reshape(-1, 1))
    # pinn_values = f(pinn_trained.to(device), torch.zeros_like(x_init).reshape(-1,1)+0.5, x_init.reshape(-1, 1))
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


    #save x to file
    with open(f"{path}/x.txt", "w") as x_file:
        for x_value in x_init:
            x_file.write(f"{x_value:.5f},")

    #save pinn values to file
    with open(f"{path}/pinn_values.txt", "w") as pinn_values_file:
        for pinn_value in pinn_values:
            pinn_values_file.write(f"{pinn_value:.5f},")


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
    with open(f"{path}/model_parameters.txt", "w") as file:
        file.write(str(pinn_trained))
    
    # write other important parameters to file
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


    if dims == 2:
        pass
        # Assuming x and t are PyTorch Tensor objects, and pinn_innit contains the solution tensor in 2D
        # Convert PyTorch Tensor to NumPy array
        # pinn_innit = pinn_values.cpu().detach().numpy()
        print(x.shape, t.shape)
        print(pinn_trained)
        pinn_values = f(pinn_trained.to(device), x, t)
        print(pinn_values.shape)
        print(pinn_values)
        # x, t = x.cpu().detach().numpy(), t.cpu().detach().numpy()
        # print(x.shape, t.shape)

        
        # # Plot the image using Matplotlib
        # plt.imshow(pinn_innit, cmap='jet', origin='lower', extent=[t.min(), t.max(), x.min(), x.max()])
        # plt.colorbar()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Solution')
        # plt.savefig(f"{path}/solution_2d.png")
        # plt.show()
