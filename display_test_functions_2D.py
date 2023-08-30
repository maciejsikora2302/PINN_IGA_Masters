import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if __name__ == '__main__':

    # files_with_display_paths = [
    #     "optimal_test_functions.txt"
    # ]

    files_with_display_paths = sys.argv[1:-1]   # these files should be saved in the display_paths folder
    folder_to_save_images    = sys.argv[-1]

    for file_with_display_path in files_with_display_paths:
        for loss_name in [
                # "loss_fn_basic",
                "loss_fn_strong",
                "loss_fn_weak",
                "loss_fn_weak_and_strong"
            ]:
            with open(file_with_display_path) as file1:
                file = file1.read()
                
                file = file.replace("\\","/")
                files = file.split('\n')
                for file in files:
                    test_function_name = os.path.join(file, loss_name, "test_functions_values.txt")
                    domain_name = os.path.join(file, loss_name, "x.txt")
                    time = os.path.join(file, loss_name, "time.txt")
                    parameters = os.path.join(file, loss_name, "other_parameters.txt")

                    with open(test_function_name, 'r') as f:
                        test_function_values = f.read()
                        test_function_values = test_function_values.replace('\n', ' ')
                        test_function_values = test_function_values.replace('[','')
                        test_function_values = test_function_values.replace(']','')
                        test_function_values = test_function_values.replace(',', ' ')
                        test_function_values = test_function_values.split()
                        test_function_values = np.array([float(num) for num in test_function_values if num != ''], dtype=float)

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
                    test_function_values = test_function_values.reshape((len(X), len(Y)))
                    print("Points shape: ", x_values.shape)
                    print("Used loss function: ", loss_name)
                    print("Time: ", t)
                    print("Epsilon: ", par)

                    # Create the figure and axes
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')

                    # Plot the 3D surface
                    ax.plot_surface(X, Y, test_function_values, cmap='viridis')

                    # Set labels and title
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('u')
                    ax.set_zlim(0,1)
                    ax.set_title('Optimal test function')

                    # Show the plot
                    plt.show()
                    print(file_with_display_path)
                    # save_path = os.path.join("C:/Users/patry/OneDrive/Pulpit/Studia_II_stopien/Magisterka/Results/", file_with_display_path[:-4], f"{loss_name}_x{x_values.shape[0]}_eps{par}_time{t}.png")
                    save_path = os.path.join(folder_to_save_images, file_with_display_path[:-4], f"{loss_name}_x{x_values.shape[0]}_eps{par}_time{t}.png")
                    fig.savefig(save_path)

