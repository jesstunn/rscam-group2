# ========================================================
# Imports
# ========================================================
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

print("I am running the Model file with Robin BC on the bottom!!!")

# ========================================================
# Dimensional Parameters
# ========================================================

# Define mesh parameters
sas_length = 2.0   # Length (L)
sas_width = 1.0    # Width (W)
nx_points = 18     # Grid points in x-direction
ny_points = 13     # Grid points in y-direction

# ========================================================
# Mesh and Function Space
# ========================================================

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(sas_length, sas_width), nx_points, ny_points)

# Define function space with linear finite elements
V = FunctionSpace(mesh, 'P', 1)  # Degree of basis polynomials = 1

# ========================================================
# Boundary Markers for Robin and Neumann Conditions
# ========================================================

# Create a MeshFunction to mark boundaries
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)  # Initialize all boundaries with marker "0"

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], sas_width) and on_boundary

# Mark boundaries with unique IDs
bottom = BottomBoundary()
top = TopBoundary()
bottom.mark(boundary_markers, 1)  # Mark bottom boundary with ID=1
top.mark(boundary_markers, 2)     # Mark top boundary with ID=2

# Use these markers with the ds measure
ds_bottom = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=1)
ds_top = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=2)

# ========================================================
# Boundary Conditions for Dirichlet BCs (Left and Right)
# ========================================================

def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

def right_boundary(x, on_boundary):
    return near(x[0], sas_length) and on_boundary

bc_left = DirichletBC(V, Constant(1.0), left_boundary)
bc_right = DirichletBC(V, Constant(0.0), right_boundary)

bcs = [bc_left, bc_right]

# ========================================================
# Variational Problem with Robin and Neumann BCs
# ========================================================

u = TrialFunction(V)
v = TestFunction(V)

D = Constant(1.0)   # Diffusion coefficient
f = Constant(0.0)   # Source term (f=0 for pure diffusion)
k = Constant(10.0)  # Robin condition coefficient for bottom boundary

# Weak form of diffusion equation:
a = D * dot(grad(u), grad(v)) * dx + k * u * v * ds_bottom  # Robin condition on bottom
L = f * v * dx                  
# Neumann condition on top is natural (no explicit term needed)

# ========================================================
# Compute Solution and Output Results
# ========================================================

u = Function(V)
solve(a == L, u, bcs)

# Plot solution and mesh
plot(u, title="Diffusion Solution with Robin BC (Bottom)")
plot(mesh, title="Mesh Grid")

plt.show()

vtkfile = File('solution_robin.pvd')
vtkfile << u
