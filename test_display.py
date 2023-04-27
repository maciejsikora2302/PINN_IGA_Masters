# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import numpy as np

# # Define the function to plot
# def f(x, y):
#     return np.sin(np.sqrt(x**2 + y**2))

# # Generate x and y data
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x, y)

# # Evaluate the function at each x, y point
# Z = f(X, Y)

# # Create a 3D plot of the function
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='coolwarm')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

def test(**kwargs):
    print(kwargs)

test(a=1, b=2, c=3)

def wr():
    return test

wr()(a=1, b=2, c=3)