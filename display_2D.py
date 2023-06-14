import numpy as np
import matplotlib.pyplot as plt
import os

files_with_display_paths = [
    "pinn_vpinn_tests_2D.txt",
    "pinn_vpinn_tests_uneven_2D.txt",
    "pinn_vpinn_tests_uneven_coef_opt.txt"
]

for file_with_display_path in files_with_display_paths:
    for loss_name in [
            "loss_fn_basic",
            "loss_fn_strong",
            "loss_fn_weak",
            "loss_fn_weak_and_strong"
        ]:
        with open(f"./display_paths/{file_with_display_path}") as file1:
            file = file1.read()
            
            file = file.replace("\\","/")
            files = file.split('\n')
            for file in files:
                pinn_values_name = os.path.join(file, loss_name, "pinn_values.txt")
                domain_name = os.path.join(file, loss_name, "x.txt")
                time = os.path.join(file, loss_name, "time.txt")
                parameters = os.path.join(file, loss_name, "other_parameters.txt")

                with open(pinn_values_name, 'r') as f:
                    pinn_values = f.read()
                    pinn_values = pinn_values.replace('\n', '')
                    pinn_values = pinn_values.replace('[','')
                    pinn_values = pinn_values.replace(']','')
                    pinn_values = pinn_values.split(',')
                    
                    pinn_values = np.array([float(num) for num in pinn_values if num != ''], dtype=float)

                with open(domain_name) as f:
                    x_values = f.read()
                    x_values = x_values.split(',')[:-1]
                    x_values = np.array([float(elem) for elem in x_values])
                    y_values = np.linspace(0,1,len(x_values))


                with open(time) as t:
                    t = t.read()
                    t = t.replace(".","_")

                with open(parameters) as par:
                    par = par.read()
                    par = par.split(",")
                    par = par[4]
                    par = par.split()[1]
                    par = par.replace(".", "")

                X, Y = np.meshgrid(x_values, y_values)
                pinn_values = pinn_values.reshape((len(X), len(Y)))
                print("Points shape: ", x_values.shape)
                print("Used loss function: ", loss_name)
                print("Time: ", t)
                print("Epsilon: ", par)

                # Create the figure and axes
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                # Plot the 3D surface
                ax.plot_surface(X, Y, pinn_values, cmap='viridis')

                # Set labels and title
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('u')
                ax.set_zlim(0,1)
                ax.set_title('The solution of the Eriksson-Johnson problem')

                # Show the plot
                plt.show()
                # file_with_display_path = file_with_display_path[:-4]
                # file_with_display_path = file_with_display_path.replace("\\", "/")
                # file_with_display_path = file_with_display_path + "/"
                print(file_with_display_path)
                save_path = os.path.join("C:/Users/patry/OneDrive/Pulpit/Studia_II_stopien/Magisterka/Results/", file_with_display_path[:-4], f"{loss_name}_x{x_values.shape[0]}_eps{par}_time{t}.png")
                fig.savefig(save_path)

