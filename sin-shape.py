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
# Create points for the lower boundary with a centered sine wave
points_lower = []
for i in range(nx + 1):
    x = i*L/ nx
    if x0 - l/2 <= x <= x0 + l/2:
        y = -h/2 * (np.sin(np.pi/2+2*np.pi * (x - x0) /l)+1)    
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

V = VectorFunctionSpace(mesh, "P", 2)  
Q = FunctionSpace(mesh, "P", 1)  
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
C = FunctionSpace(mesh, "CG", 1)

inflow = Expression(("4.0*x[1]*(1.0-x[1])", "0.0"), degree=2)  # Poiseuille
noslip = Constant((0.0, 0.0))  
outflow = Constant(0.0)  

def right(x, on_boundary): 
    return near(x[0], L, DOLFIN_EPS) and on_boundary

def left(x, on_boundary): 
    return near(x[0], 0.0, DOLFIN_EPS) and on_boundary

def bottom(x, on_boundary):  
    return on_boundary and near(x[1], min(p[1] for p in points_lower), DOLFIN_EPS)

def top(x, on_boundary): 
    return near(x[1], 1.0, DOLFIN_EPS) and on_boundary

# Boundary condition
bc_inlet = DirichletBC(W.sub(0), inflow, left)  
bc_noslip_bottom = DirichletBC(W.sub(0), noslip, bottom)  
bc_noslip_top = DirichletBC(W.sub(0), noslip, top)  
bc_outlet = DirichletBC(W.sub(1), outflow, right)  

bcs = [bc_inlet, bc_noslip_bottom, bc_noslip_top, bc_outlet]

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
n = Constant((0.0,1.0))
S = Constant((0.0,0.0))
a = inner(grad(u)+grad(u).T, grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx + inner(S,v)*ds

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

solver = KrylovSolver("minres", "amg")
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()
ux, uy = u.split()

# Advection-diffusion Problem
bc5_c = DirichletBC(C, Constant(0.0), left) 

c_sol = TrialFunction(C)
phi = TestFunction(C)
D = Constant(0.05)
a_c = (D * inner(grad(c_sol), grad(phi)) + inner(dot(u, grad(c_sol)), phi)) * dx
L_c = Constant(0) * phi * dx

c_sol = Function(C)
solve(a_c == L_c, c_sol, bc5_c)

File("solution_P.pvd") << u
File("solute_concentration_P.pvd") << c_sol



