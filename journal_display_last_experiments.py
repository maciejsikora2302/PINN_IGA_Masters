import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from pprint import pprint
import torch
import json
from display_multiple import modify_loss_values, add_to_loss_plot_moded
torch.set_default_dtype(torch.float64)

SAVE_PATH = "./display_imgs/publication_imgs/extra_experiments"

eps_10_times_smaller = [
    "out_data/2023-12-07_23-17-23",
    "out_data/2023-12-08_01-10-27"
]


eps_100_times_smaller = [
    "out_data/2023-12-07_23-17-32",
    "out_data/2023-12-08_01-15-58"
]

eps_10_times_larger = [
    "out_data/2023-12-07_23-17-54",
    "out_data/2023-12-08_01-09-15"
]

function_names = [
    "loss_fn_basic",
    "loss_fn_strong",
    "loss_fn_weak",
    "loss_fn_weak_and_strong",
]

labels = {
    "loss_fn_basic": "PINN",
    "loss_fn_strong": "vPINN Strong",
    "loss_fn_weak": "vPINN Weak",
    "loss_fn_weak_and_strong": "vPINN Weak and Strong",
}

line_styles = [(0, (1, 1, 10, 1)), '--', '-.', ':', (0, (1, 10)), (0, (1, 1)), 
               (0, (5, 10)), (0, (5, 5)), (0, (5, 1, 1, 1))]

fig_size = (21, 14)

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

def gather_values(arr_save, current_arr):
    if len(arr_save) == 0:
        arr_save = current_arr
    else:
        assert len(arr_save) == len(current_arr)
        for i, val in enumerate(current_arr):
            arr_save[i] += val
    return arr_save

def create_3x2_grid():
    fig, axs = plt.subplots(3, 2, constrained_layout = True, figsize=fig_size)
    
    return fig, axs


def add_to_loss_plot_raw(ax, loss_values, label, other_parameters, line_style="-", linewidth=2):
    #plot in log scale
    loss_values = np.array(loss_values)
    # loss_values = np.log10(loss_values)
    ax.plot(loss_values, label=label, linestyle=line_style, linewidth=linewidth)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # upper_lim = max(loss_values)+1
    # lower_lim = min(loss_values)-1
    # if lower_lim < -10:
    #     lower_lim = -10
    # ax.set_ylim(lower_lim, upper_lim)
    ax.legend(fontsize=14)
    ax.grid()
    title = f"\
    lr={other_parameters['learning_rate']}".replace("    ", "")
    ax.title.set_text(title)

def add_to_solution_plot(ax, data, values, label, other_parameters, time, line_style="-", linewidth=2):
    ax.plot(data, values, label=label, linestyle=line_style, linewidth=linewidth)
    ax.set_xlabel("X")
    ax.set_ylabel("Values")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=14)
    # plt.legend(prop = {'size' : 1})
    ax.grid()
    title = f"\
    lr={other_parameters['learning_rate']} \
    t[s]={time}".replace("    ", "")
    ax.title.set_text(title)

plt.rcParams.update({'font.size': 19})
matplotlib.rcParams['figure.figsize'] = fig_size
matplotlib.rcParams['figure.dpi'] = 300


plt.rcParams.update({'font.size': 19})

fig_loss, axs_loss = create_3x2_grid()
fig_loss_raw, axs_loss_raw = create_3x2_grid()
fig_solution, axs_solution = create_3x2_grid()

axs_loss, axs_loss_raw, axs_solution = axs_loss.flatten(), axs_loss_raw.flatten(), axs_solution.flatten()
# fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

for i, experiment in enumerate(eps_10_times_smaller + eps_100_times_smaller + eps_10_times_larger):
    for next_style, function_name in enumerate(function_names):
        file_path = experiment
        pinn_values = load_values(os.path.join(file_path, function_name, "pinn_values.txt"))
        loss_values = load_values(os.path.join(file_path, function_name, "loss_values.txt"))
        other_parameters = load_other_parameters(os.path.join(file_path, function_name))
        x = load_values(os.path.join(file_path, function_name, "x.txt"))
        # ax, loss_values, label, other_parameters, line_style="-", linewidth=2
        add_to_loss_plot_raw(axs_loss_raw[i], loss_values, f"{labels[function_name]}", other_parameters=other_parameters, line_style=line_styles[next_style], linewidth=2)
        # # ax, loss_values, label, other_parameters, line_style="-", linewidth=2
        # add_to_loss_plot_moded(axs_loss[i], loss_values, "", other_parameters=other_parameters, line_style=line_styles[i], linewidth=2)
        # ax, data, values, label, other_parameters, time, line_style="-", linewidth=2
        add_to_solution_plot(axs_solution[i], x, pinn_values, f"{labels[function_name]}" ,other_parameters=other_parameters,  time = 0, line_style=line_styles[next_style], linewidth=2)


print("Saving plots")
fig_loss.savefig(os.path.join(SAVE_PATH, "loss_plot.png"))
fig_loss_raw.savefig(os.path.join(SAVE_PATH, "loss_plot_raw.png"))
fig_solution.savefig(os.path.join(SAVE_PATH, "solution_plot.png"))