import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
from general_parameters import OUT_DATA_NAME

argparser = argparse.ArgumentParser()
argparser.add_argument('-f','--folder', type=str, default=None, help='Folder to display')
args = argparser.parse_args()

given_folder = args.folder

if given_folder is not None and os.path.exists(given_folder):
    list_of_functions = glob.glob(f'{given_folder}\*')
else:
    list_of_files = glob.glob(f'.\{OUT_DATA_NAME}\*')
    # print(list_of_files)
    latest_folder = max(list_of_files, key=os.path.getctime)
    print(f"Latest folder: {latest_folder}")

    list_of_functions = glob.glob(f'{latest_folder}\*')
# print(list_of_functions)


data = []
for function in list_of_functions:
    with open(f"{function}/solution_profile.txt") as file:
        lines = file.readlines()
        splitted = lines[0].replace('tensor','').replace('(','').replace(')','').replace('\n','').split(',')
        x = list(map(lambda x: float(x), splitted))
        splitted = lines[1].replace('tensor','').replace('(','').replace(')','').replace('\n','').split(',')
        splitted = list(filter(lambda x: x != ' device=\'cuda:0\'', splitted))
        # print(splitted)
        y = list(map(lambda x: float(x), splitted))
        data.append((x, y))

#display four graphs
fig, axs = plt.subplots(2, 2)
fig.suptitle('Solution profiles')
axs[0, 0].plot(data[0][0], data[0][1])
loss_name = ' '.join(list_of_functions[0].split("\\")[-1].split("_")[2:])
axs[0, 0].set_title(f'Loss function {loss_name}')

try:
    axs[0, 1].plot(data[1][0], data[1][1])
    loss_name = ' '.join(list_of_functions[1].split("\\")[-1].split("_")[2:])
    axs[0, 1].set_title(f'Loss function {loss_name}')
except Exception as e:
    pass
try:
    axs[1, 0].plot(data[2][0], data[2][1])
    loss_name = ' '.join(list_of_functions[2].split("\\")[-1].split("_")[2:])
    axs[1, 0].set_title(f'Loss function {loss_name}')
except Exception as e:
    pass
try:
    axs[1, 1].plot(data[3][0], data[3][1])
    loss_name = ' '.join(list_of_functions[3].split("\\")[-1].split("_")[2:])
    axs[1, 1].set_title(f'Loss function {loss_name}')
except Exception as e:
    pass
plt.show()
