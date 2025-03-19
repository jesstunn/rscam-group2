from dolfin import *
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

p0 = Point((0.0,0.0))
p1 = Point((0.0,1.0))
p2 = Point((10,1.0))
p3 = Point((10,0.0))

domain_vertices = [p0,p3,p2,p1]
domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
C = FunctionSpace(mesh, "CG", 1)


def right(x, on_boundary): 
    return x[0] > (10.0 - 10*DOLFIN_EPS)
def left(x, on_boundary): 
    return x[0] < DOLFIN_EPS
def bottom(x, on_boundary): 
    return x[1] < 0.0 + 10.0*DOLFIN_EPS # y = 0
def top(x, on_boundary): 
    return x[1] > 1.0 - 10.0*DOLFIN_EPS # y = 1

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
noslip2 = Constant((0.0))
outflow = Expression(("0.0","-pow(x[0],2)+100"),degree=2)
inflow = Expression(("x[1]*(1 - x[1])", "0.0"), degree = 2)

bc1 = DirichletBC(W.sub(1), Constant(0.0), right) 
bc2 = DirichletBC(W.sub(0), noslip, bottom) 
bc3 = DirichletBC(W.sub(0), inflow, left)
bc0 = DirichletBC(W.sub(0), noslip, top) 


# Collect boundary conditions
bcs = [bc0, bc1, bc2, bc3]

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

bc1_c = DirichletBC(C, Constant(0.0), right) 
bc5_c = DirichletBC(C, Constant(1.0), left) 

c_sol = TrialFunction(C)
phi = TestFunction(C)
D = Constant(5)
a_c = (D * inner(grad(c_sol), grad(phi)) + inner(dot(u, grad(c_sol)), phi)) * dx
L_c = Constant(0) * phi * dx

c_sol = Function(C)
solve(a_c == L_c, c_sol, [bc1_c, bc5_c])

File("solution_2.pvd") << u
File("solute_concentration_2.pvd") << c_sol

