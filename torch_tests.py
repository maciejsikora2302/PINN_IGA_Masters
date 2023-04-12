import numpy as np
from scipy.interpolate import splev

# Define a set of control points and knot vector for a cubic B-spline
control_points = [(0, 0), (0, 1, 0, 1, .5), (2, -1, 3), (3, 0)]
knots = [0, 0, 0, 1, 2, 3, 3, 3]
knots = np.array(knots)/3.0

# Evaluate the B-spline at a set of points
t = np.linspace(0, 1, 101)
out = splev(t, (knots, control_points, 3), der=1)

print(out)

# Plot the B-spline curve
import matplotlib.pyplot as plt
plt.plot(t, out[0], 'b', lw=2, label=f'B-spline curve, control points {control_points[0]}')
plt.plot(t, out[1], 'r', lw=2, label=f'B-spline curve, control points {control_points[1]}')
plt.plot(t, out[2], 'g', lw=2, label=f'B-spline curve, control points {control_points[2]}')
plt.grid()
plt.legend()
plt.show()
