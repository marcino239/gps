# simulation of third order system

import casadi as ca 
import numpy as NP
import matplotlib.pyplot as plt
from operator import itemgetter

nk = 20    # Control discretization
tf = 10.0  # End time

# Declare variables (use scalar graph)
u  = ca.SX.sym("u")    # control
x  = ca.SX.sym("x",2)  # states

# ODE right hand side and quadratures
xdot = ca.vertcat( [(1 - x[1]*x[1])*x[0] - x[1] + u, x[0]] )
qdot = x[0]*x[0] + x[1]*x[1] + u*u

# DAE residual function
dae = ca.SXFunction("dae", ca.daeIn(x=x, p=u), ca.daeOut(ode=xdot, quad=qdot))

# Create an integrator
integrator = ca.Integrator("integrator", "cvodes", dae, {"tf":tf/nk})

# All controls (use matrix graph)
x = ca.MX.sym("x",nk) # nk-by-1 symbolic variable
U = ca.vertsplit(x) # cheaper than x[0], x[1], ...

# The initial state (x_0=0, x_1=1)
X  = ca.MX([0,1])

# Objective function
f = 0

# Build a graph of integrator calls
for k in range(nk):
  X,QF = itemgetter('xf','qf')(integrator({'x0':X,'p':U[k]}))
  f += QF

# Terminal constraints: x_0(T)=x_1(T)=0
g = X

# Allocate an NLP solver
opts = {'linear_solver': 'ma27'}
nlp = ca.MXFunction("nlp", ca.nlpIn(x=x), ca.nlpOut(f=f,g=g))
solver = ca.NlpSolver("solver", "ipopt", nlp, opts)

# Solve the problem
sol = solver({"lbx" : -0.75,
              "ubx" : 1,
              "x0" : 0,
              "lbg" : 0,
              "ubg" : 0})
              
# Retrieve the solution
u_opt = NP.array(sol["x"])
print( sol )

# Time grid
tgrid_x = NP.linspace(0,10,nk+1)
tgrid_u = NP.linspace(0,10,nk)

# Plot the results
plt.figure(1)
plt.clf()
plt.plot(tgrid_u,u_opt,'b-')
plt.title("Van der Pol optimization - single shooting")
plt.xlabel('time')
plt.legend(['u trajectory'])
plt.grid()
plt.show()

