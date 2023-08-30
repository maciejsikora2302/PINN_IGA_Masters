import numpy as np
import torch
import os
import sys

def u(x: torch.float, y: torch.float, epsilon: torch.float=0.01):

  if y == 0 or y == 1:
    return 0
  if x == 1:
    return 0
  if x == 0:
    return np.sin(np.pi * y)
  
  else:
    r_1 = (1 + np.sqrt(1 + 4 * epsilon**2 * np.pi**2)) / (2 * epsilon)
    r_2 = (1 - np.sqrt(1 + 4 * epsilon**2 * np.pi**2)) / (2 * epsilon)
    result = (np.exp(r_1 * (x-1)) - np.exp(r_2 * (x-1))) / (np.exp(-r_1) - np.exp(-r_2)) * np.sin(np.pi * y)

    return result
  

if __name__ == '__main__':

    files_with_display_paths  = sys.argv[1:-1]  # these files should be saved in the display_paths folder
    file_with_metrics_results = sys.argv[-1]    # these files should be saved in the metrics_data folder

    for file_with_display_path in files_with_display_paths:
        for loss_name in [
                "loss_fn_basic",
                "loss_fn_strong",
                "loss_fn_weak",
                "loss_fn_weak_and_strong"
            ]:
            with open(file_with_display_path) as file1:
                file = file1.read()
                
                file = file.replace("\\","/")
                files = file.split('\n')
                for file in files:
                    pinn_values_name = os.path.join(file, loss_name, "pinn_values.txt")
                    domain_name = os.path.join(file, loss_name, "x.txt")
                    time = os.path.join(file, loss_name, "time.txt")
                    parameters = os.path.join(file, loss_name, "other_parameters.txt")
                    other_parameters = os.path.join(file, loss_name, "other_parameters.txt")

                    with open(pinn_values_name, 'r') as f:
                        pinn_values = f.read()
                        pinn_values = pinn_values.replace('\n', ' ')
                        pinn_values = pinn_values.replace('[','')
                        pinn_values = pinn_values.replace(']','')
                        pinn_values = pinn_values.replace(',', ' ')
                        pinn_values = pinn_values.split()
                        pinn_values = np.array([float(num) for num in pinn_values if num != ''], dtype=float)

                    with open(domain_name) as f:
                        x_values = f.read()
                        x_values = x_values.split(',')[:-1]
                        x_values = np.array([float(elem) for elem in x_values])
                        y_values = np.linspace(0,1,len(x_values))

                    with open(parameters) as par:
                        par = par.read()
                        par = par.split(",")
                        par = par[4]
                        par = par.split()[1]
                        par = par.replace(".", "")
                    
                    with open(other_parameters) as other_par:
                       other_par = other_par.read()
                       other_par = other_par.split(",")[-1]
                       other_par = other_par.split(":")[-1][:-1]

                    n_points = len(x_values)

                    result = []
                    
                    for x in x_values:
                        for y in y_values:
                            result.append(u(x,y, epsilon=np.float64(par)))
                    
                    result = np.array(result)

                    sum_diff_squared = np.sum((result - pinn_values)**2)
                    sum_exact_sol = np.sum(result**2) 
                    accuracy = sum_diff_squared / sum_exact_sol
                    accuracy = np.sqrt(accuracy)

                    # if loss_name == "loss_fn_weak_and_strong" and "true" in other_par:
                    print("Points shape: ", f"{x_values.shape[0]}^2")
                    print("Used loss function: ", loss_name)
                    print("Epsilon: ", f"0.{par[1:]}")
                    print("Uneven: ", other_par)
                    print("Accuracy: ", round(accuracy, 3))
                    print("####################################################")

                    if "true" in other_par:
                        with open(file_with_metrics_results, "+a") as metrics_file:
                            metrics_file.write("####################################################\n")
                            metrics_file.write(f"Points shape: {x_values.shape[0]}^2\n")
                            metrics_file.write(f"Used loss function: {loss_name}\n")
                            metrics_file.write(f"Epsilon: 0.{par[1:]}\n")
                            metrics_file.write(f"Uneven: {other_par}\n")
                            metrics_file.write(f"Metric measure: {accuracy:.3f}\n")
                            