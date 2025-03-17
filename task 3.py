from dolfin import *
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

p0 = Point((0.0,0.0))
p1 = Point((0.0,-1.0))
p2 = Point((10,-1.0))
p3 = Point((10,0.0))

domain_vertices = [p0,p1,p2,p3]
domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))


def right(x, on_boundary): 
    return x[0] > (10.0 - 10*DOLFIN_EPS)
def left(x, on_boundary): 
    return x[0] < DOLFIN_EPS
def bottom(x, on_boundary): 
    return x[1] < -1.0 + 10.0*DOLFIN_EPS
def top(x, on_boundary): 
    return x[1] > - 10.0*DOLFIN_EPS

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
noslip2 = Constant((0.0))
outflow = Expression(("0.0","-pow(x[0],2)+100"),degree=2)
inflow = Expression(("x[1] = x[1]*(x[1]-1)", "0.0"), degree = 2)

bc1 = DirichletBC(W.sub(1), Constant(0.0), right) 
bc2 = DirichletBC(W.sub(0), noslip, bottom) 
bc5 = DirichletBC(W.sub(0).sub(1), Constant(0.0), right) 
bc3 = DirichletBC(W.sub(0).sub(0), inflow, left) 
bc0 = DirichletBC(W.sub(0), outflow, top) 


# Collect boundary conditions
bcs = [bc0, bc1, bc2, bc3, bc5]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
n = Constant((0.0,1.0))
S = Constant((0.0,0.0))
a = inner(grad(u)+grad(u).T, grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx + inner(S,v)*ds

b = inner(grad(u), grad(v))*dx + p*q*dx

u, p = U.split()
ux, uy = u.split()

tol = 0.001
x = np.linspace(0 + tol, 1 - tol, 101)
points = [(x_, 0) for x_ in x]
u_line = np.array([ux(point) for point in points])

def func(x,alpha,beta):
    return alpha*(x*(1.0-x))**beta

bopt, bcov = curve_fit(func,x,u_line)


print(bopt)
print(bcov)

ufile_pvd = File("stokes/velocity.pvd")
ufile_pvd << u
ufile_pvd = File("stokes/velocityx.pvd")
ufile_pvd << ux
ufile_pvd = File("stokes/velocityy.pvd")
ufile_pvd << uy
pfile_pvd = File("stokes/pressure.pvd")
pfile_pvd << p

plot(u)
plt.show()