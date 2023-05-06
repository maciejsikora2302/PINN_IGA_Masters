import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pprint import pprint as pp
import torch
import json
from PINN import PINN, get_activation_function
# Paths will be passed in from the command line
# Get all paths to the files and validate them

def get_paths(path_to_file_containig_paths: str):
    if not os.path.exists(path_to_file_containig_paths):
        raise ValueError(f"Path {path_to_file_containig_paths} does not exist")
    
    with open(path_to_file_containig_paths, "r") as f:
        paths = f.readlines()
    
    paths = list(map(lambda x: x.replace("\n", ""), paths))
    paths = list(filter(lambda x: os.path.exists(x), paths))

    return paths

def load_model(path:str):
    model_params = json.load(open(f"{path}/model_parameters.txt", "r"))
    model = PINN(
        num_hidden=model_params["num_hidden"],
        dim_hidden=model_params["dim_hidden"],
        act=get_activation_function(activation_name=model_params["act"]),
        pinning=model_params["pinning"],
        dims=model_params["dims"],
        dim_layer_in=model_params["dim_layer_in"],
        dim_layer_out=model_params["dim_layer_out"],
        pinn_learns_coeff=model_params["pinn_learns_coeff"],
    )
    model.load_state_dict(torch.load(f"{path}/model.pt"))
    return model

def load_data(path:str):
    with open(path, "r") as f:
        data = f.read()
        data = data.split(",")
        if data[-1] == "":
            data = data[:-1]
        data = list(map(lambda x: float(x), data))
        data = np.array(data)
    return data

def get_all_used_functions(path:str):
    function_names = []
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            function_names.append(file)
    return function_names
            
def get_values_from_model_1d(model, data):
    if data is not torch.Tensor:
        data = torch.Tensor(data)
    data = data.reshape(-1, 1)
    values = model(data, None)
    return values

def load_values(path):
    with open(path, "r") as f:
        values = f.read()
        values = values.split(",")
        if values[-1] == "":
            values = values[:-1]
        values = list(map(lambda x: float(x), values))
        values = np.array(values)
    return values

def load_other_parameters(path):
    with open(os.path.join(path, "other_parameters.txt"), "r") as f:
        other_parameters = json.load(f)
    return other_parameters

def load_time(path):
    with open(os.path.join(path, "time.txt"), "r") as f:
        time = f.read()
    return time

def add_to_loss_plot(ax, loss_values, label, other_parameters):
    #plot in log scale
    loss_values = np.log10(loss_values)
    ax.plot(loss_values, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # upper_lim = max(loss_values)+1
    # lower_lim = min(loss_values)-1
    # if lower_lim < -10:
    #     lower_lim = -10
    # ax.set_ylim(lower_lim, upper_lim)
    ax.legend()
    ax.grid()
    title = f"\
    eps={other_parameters['eps_interior']} \
    epochs={other_parameters['epochs']} \
    X={other_parameters['n_points_x']}".replace("    ", "")
    ax.title.set_text(title)

def add_to_solution_plot(ax, data, values, label, other_parameters,time):
    ax.plot(data, values, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Values")
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid()
    title = f"\
    eps={other_parameters['eps_interior']} \
    epochs={other_parameters['epochs']} \
    X={other_parameters['n_points_x']} \
    t[s]={time}".replace("    ", "")
    ax.title.set_text(title)

def create_3x3_grid():
    fig, axs = plt.subplots(3, 3, constrained_layout = True)
    # fig.set_size_inches(18.5, 10.5)
    # fig.tight_layout(pad=2.5)
    
    return fig, axs

if __name__ == "__main__":
    paths = get_paths(sys.argv[1])

    if len(paths) == 0:
        raise ValueError("No valid paths found")
    if any(map(lambda x: not os.path.isdir(x), paths)):
        raise ValueError("All paths must be directories")
    if any(map(lambda x: not os.path.exists(x), paths)):
        raise ValueError("All paths must exist")
    functions = get_all_used_functions(paths[0])
    fig_loss, axs_loss = create_3x3_grid()
    fig_solution, axs_solution = create_3x3_grid()

    for function in functions:
        # fig_loss.suptitle(f"Loss values for {function} in log10 scale")
        fig_loss.suptitle(f"Loss values in log10 scale")
        fig_solution.suptitle(f"Solution profile normalized to [0,1]")

        if len(paths) == 6:
            # 3x2 grid by removing last column
            axs_loss = axs_loss[:, :-1]
            axs_solution = axs_solution[:, :-1]


        for path, ax_loss, ax_solution in zip(paths, axs_loss.flatten(), axs_solution.flatten()):
            model = load_model(os.path.join(path, function))
            data = load_data(os.path.join(path, function, "x.txt"))
            # values = get_values_from_model_1d(model, data)
            pinn_values = load_values(os.path.join(path, function, "pinn_values.txt"))
            loss_values = load_values(os.path.join(path, function, "loss_values.txt"))
            other_parameters = load_other_parameters(os.path.join(path, function))
            time = load_time(os.path.join(path, function))
            add_to_loss_plot(ax_loss, loss_values, function, other_parameters)
            add_to_solution_plot(ax_solution, data, pinn_values, function, other_parameters, time)

    plt.show()
        
