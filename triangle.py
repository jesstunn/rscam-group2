from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 200
ny = 50
h = 0.5; # thickness of sulcus
L=2.; # length of the domain
l = L*0.3; # length of the sulcus

x0 = L/2  # center point
print(x0+l/2)
points_lower = []
# Create points for the lower boundary with a triangle
for i in range(nx + 1):
    x = i * L / nx 
    if x0 - l/2 <= x <= x0:  
        y = -5/3*(x - (x0 - l/2)) 
    elif x0 < x <= x0 + l/2: 
        y = -h + (5/3)*(x - x0) 
    else:
        y = 0  

points_lower.append((x, y))

# Define points for the upper boundary and sides
points_upper = [(i / nx*L, 1) for i in range(nx + 1)]
points_left = [(0, j / ny) for j in range(1, ny)]
points_right = [(L, j / ny) for j in range(1, ny)]

# Combine all points
points = points_lower + points_right + points_upper[::-1] + points_left[::-1]

# Define the polygon
polygon = Polygon([Point(x, y) for x, y in points])

# Create the mesh from the polygon
mesh = generate_mesh(polygon, 50)

# Plot the mesh
plot(mesh)
plt.show()