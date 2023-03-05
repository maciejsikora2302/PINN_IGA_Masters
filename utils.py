import torch
import os
import numpy as np
from general_parameters import logger, Color, OUT_DATA_FOLDER
from loss_functions import compute_loss
import matplotlib.pyplot as plt
from NN_tools import f
from plotting import plot_color

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def compute_losses_and_plot_solution(pinn_trained, x, device, loss_values, x_init, u_init, N_POINTS_X, N_POINTS_T, loss_fn_name, t=None, dims:int = 2):
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

    # logger.info(f"Computing final loss")

    # losses = compute_loss(pinn_trained.to(device), x=x, t=t, verbose=True, dims=dims)
    # logger.info(f"{'Total loss:':<50}{Color.GREEN}{losses[0]:.5f}{Color.RESET}    ({losses[0]:.3E})")
    # logger.info(f"{'Interior loss:':<50}{Color.GREEN}{losses[1]:.5f}{Color.RESET}    ({losses[1]:.3E})")
    # logger.info(f"{'Initial loss:':<50}{Color.GREEN}{losses[2]:.5f}{Color.RESET}    ({losses[2]:.3E})")
    # logger.info(f"{'Bondary loss:':<50}{Color.GREEN}{losses[3]:.5f}{Color.RESET}    ({losses[3]:.3E})")

    average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Loss function (runnig average)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)

    plt.savefig(f"{path}/loss.png")

    # z = f(pinn_trained.to(device), x, t)
    # color = plot_color(z.cpu(), x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T)
    # plt.plot(x_init, u_init, label="Initial condition")
    # plt.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
    # plt.legend()
    # plt.savefig("./imgs/initial_condition2.png")

    if device == 'cuda':
        pinn_init = f(pinn_trained.cuda(), x_init.reshape(-1, 1), torch.zeros_like(x_init).reshape(-1,1))
    else:
        pinn_init = f(pinn_trained.cpu(), x_init.reshape(-1, 1), torch.zeros_like(x_init).reshape(-1,1))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Initial condition difference")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.plot(x_init.cpu(), u_init, label="Initial condition")
    ax.plot(x_init.cpu(), pinn_init.cpu().flatten().detach(), label="PINN solution")
    ax.legend()
    plt.savefig(f"{path}/initial_condition.png")

    # from IPython.display import HTML
    # ani = plot_solution(pinn_trained.cpu(), x.cpu(), t.cpu())
    # HTML(ani.to_html5_video())

    # plt.plot(x_init, u_init, label="Initial condition")
    # plt.plot(x_init, pinn_init.flatten().detach(), label="PINN solution")
    # plt.legend()

    pinn_init = f(pinn_trained.cuda(), torch.zeros_like(x_init).reshape(-1,1)+0.5, x_init.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    ax.set_title("Solution profile")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.plot(x_init, pinn_init.cpu().flatten().detach(), label="PINN solution")
    ax.legend()
    plt.savefig(f"{path}/solution_profile.png")
    with open(f"{path}/solution_profile.txt", "w") as file:
        file.write(','.join([str(x) for x in x_init]))
        file.write('\n')
        y = pinn_init.flatten().detach()
        file.write(','.join([str(x) for x in y]))