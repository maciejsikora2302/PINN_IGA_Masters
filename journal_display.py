import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from pprint import pprint
import torch
import random
torch.set_default_dtype(torch.float64)

SAVE_PATH = "./display_imgs/publication_imgs"

first_iteration = [
    "out_data/2023-11-20_21-43-22",
    "out_data/2023-11-20_23-49-21",
    "out_data/2023-11-21_01-38-46",
    "out_data/2023-11-21_03-35-26",
    "out_data/2023-11-21_05-31-13",
    "out_data/2023-11-21_07-27-08",
    "out_data/2023-11-21_09-21-04",
    "out_data/2023-11-21_11-17-34",
    "out_data/2023-11-21_13-11-27",
    "out_data/2023-11-21_15-06-18",
]

second_iteration_paths = [
    "out_data/2023-11-21_16-59-35",
    "out_data/2023-11-21_18-57-17",
    "out_data/2023-11-21_20-53-22",
    "out_data/2023-11-21_22-53-15",
    "out_data/2023-11-22_01-57-50",
    "out_data/2023-11-22_03-51-13",
    "out_data/2023-11-22_05-39-37",
    "out_data/2023-11-22_07-26-14",
    "out_data/2023-11-22_09-21-07",
    "out_data/2023-11-22_11-13-04",
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

sums_of_pinns = {
    "loss_fn_basic": [],
    "loss_fn_strong": [],
    "loss_fn_weak": [],
    "loss_fn_weak_and_strong": [],
}

line_styles = ['-', '--', '-.', ':', (0, (1, 10)), (0, (1, 1)), 
               (0, (1, 1, 10, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1, 1, 1))]



def load_values(path):
    with open(path, "r") as f:
        values = f.read()
        values = values.split(",")
        if values[-1] == "":
            values = values[:-1]
        values = list(map(lambda x: float(x), values))
        values = np.array(values)
    return values

def gather_values(arr_save, current_arr):
    if len(arr_save) == 0:
        arr_save = current_arr
    else:
        assert len(arr_save) == len(current_arr)
        for i, val in enumerate(current_arr):
            arr_save[i] += val
    return arr_save

plt.rcParams.update({'font.size': 19})
matplotlib.rcParams['figure.figsize'] = (16, 12)
matplotlib.rcParams['figure.dpi'] = 300

# fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

for iteration in [first_iteration, second_iteration_paths]:
    for function_name in function_names:
        for i, path in enumerate(iteration):
            pinn_values = load_values(os.path.join(path, function_name, "pinn_values.txt"))
            x = load_values(os.path.join(path, function_name, "x.txt"))
            plt.plot(x, pinn_values, label=f"Iteration {i+1}", linestyle=line_styles[i], linewidth=2)
            plt.xlabel("x")
            plt.ylabel("Values")
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.title(f"Aggregation of {labels[function_name]} ({function_name})")


        preffix = "even" if iteration == first_iteration else "uneven"
        preffix += "_dist"
        plt.grid()
        plt.savefig(os.path.join(SAVE_PATH, f"{preffix}_aggregation_{function_name}.png"))
        plt.clf()
        plt.cla()

for iteration in [first_iteration, second_iteration_paths]:
    for i, path in enumerate(iteration):
        for function_name in function_names:
            pinn_values = load_values(os.path.join(path, function_name, "pinn_values.txt"))
            sums_of_pinns[function_name] = gather_values(sums_of_pinns[function_name], pinn_values)
    
    for function_name in function_names:
        # print(len(iteration))
        # pprint(sums_of_pinns[function_name][:10]) if iteration == first_iteration else None
        # pprint(sums_of_pinns[function_name]) if iteration == first_iteration else None
        sums_of_pinns[function_name] = sums_of_pinns[function_name] / len(iteration)
        x = load_values(os.path.join(path, function_name, "x.txt"))
        plt.plot(x, sums_of_pinns[function_name], label = f"All {labels[function_name]}", linewidth=2, linestyle=line_styles[random.choice(range(len(line_styles)))])

    
    plt.xlabel("x")
    plt.ylabel("Values")
    # plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()
    title_part = 'even' if iteration == first_iteration else 'uneven'
    plt.title(f"Averages of {title_part} distributions")
    plt.savefig(os.path.join(SAVE_PATH, f"all_aggregation_{title_part}.png"))
    plt.clf()
    plt.cla()

    for key in sums_of_pinns.keys():
        sums_of_pinns[key] = []

    #     #plot avg of sums of pinns
    #     for i, val in enumerate(sums_of_pinns[function_name]):
    #         sums_of_pinns[function_name][i] = val / len(first_iteration)

    #     plt.plot(x, sums_of_pinns[function_name], label=f"Iteration {i+1}", linestyle=line_styles[i], linewidth=2)
    #     plt.xlabel("x")
    #     plt.ylabel("Values")
    #     plt.ylim(-0.1, 1.1)
    #     plt.legend()
    #     plt.title(f"Aggregation of {labels[function_name]} ({function_name})")
        


    # #clear sums of pinns after changing iteration
    # sums_of_pinns = {
    #     "loss_fn_basic": [],
    #     "loss_fn_strong": [],
    #     "loss_fn_weak": [],
    #     "loss_fn_weak_and_strong": [],
    # }
