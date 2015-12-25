#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

nk = 50      # Control discretization
tf = 20.0    # End time
coll = False # Use collocation integrator

# initial and target points
x0       = ca.DMatrix( [0, np.pi/3, 0, 0, 0 ] )
x_target = ca.DMatrix( [0, 0      , 0, 0, 0 ] )

# control limitations
u_min = -40.
u_max =  40.
x_min = -10.
x_max =  10.

# constants
g     = 9.81
mc    = 1.0
mp    = 1.
l     = 1.
mip   = 1.
mic   = 5.

# Declare variables (use scalar graph)
t  = ca.SX.sym( 't' )      # time
u  = ca.SX.sym( 'u' )      # control
x  = ca.SX.sym( 'x' , 5 )  # state

# ODE rhs function
#   x[0] = dtheta
#   x[1] = theta
#   x[2] = dx
#   x[3] = x
#   x[4] = cost

sint = ca.sin( x[1] )
cost = ca.cos( x[1] )

fric = mic * ca.sign( x[2] )
num  = g * sint + cost * ( -u - mp * l * x[0] * x[0] * sint + fric ) / ( mc + mp ) - mip * x[0] / ( mp * l )
denom = l * ( 4. / 3. - mp * cost*cost / (mc + mp) )
ddtheta = num / denom
ddx = (-u + mp * l * ( x[0] * x[0] * sint - ddtheta * cost ) - fric) / (mc + mp)

x_err = x-x_target
cost_mat = np.diag( [1,1,1,1,0] )

ode = ca.vertcat([	ddtheta, \
					x[0], \
					ddx, \
					x[2], \
					ca.mul( [ x_err.T, cost_mat, x_err ] )
				] )

dae = ca.SXFunction( 'dae', ca.daeIn( x=x, p=u, t=t ), ca.daeOut( ode=ode ) )

# Create an integrator
opts = { 'tf': tf/nk } # final time
if coll:
  opts[ 'number_of_finite_elements' ] = 5
  opts['interpolation_order'] = 5
  opts['collocation_scheme'] = 'legendre'
  opts['implicit_solver'] = 'kinsol'
  opts['implicit_solver_options'] =  {'linear_solver' : 'csparse'}
  opts['expand_f'] = True
  integrator = ca.Integrator('integrator', 'oldcollocation', dae, opts)
else:
  opts['abstol'] = 1e0 # tolerance
  opts['reltol'] = 1e0 # tolerance
#  opts['steps_per_checkpoint'] = 1000
  opts['quad_err_con'] = True
  opts['fsens_err_con' ] = True 
#  opts['t0'] = 0.
#  opts['tf'] = tf
  integrator = ca.Integrator('integrator', 'cvodes', dae, opts)

# Total number of variables
nv = u.shape[0] * nk + x.shape[0] * (nk+1)

# Declare variable vector
V = ca.MX.sym( 'V', nv )

# Get the expressions for local variables
U  = V[    0:nk   ]
X0 = V[ nk+0:nv:5 ]
X1 = V[ nk+1:nv:5 ]
X2 = V[ nk+2:nv:5 ]
X3 = V[ nk+3:nv:5 ]
X4 = V[ nk+4:nv:5 ]


# Variable bounds initialized to +/- inf
VMIN = -np.inf*np.ones(nv)
VMAX =  np.inf*np.ones(nv)

# Control bounds
VMIN[0:nk] = u_min
VMAX[0:nk] = u_max

# Initial condition
VMIN[nk+0] = VMAX[nk+0] = x0[0]
VMIN[nk+1] = VMAX[nk+1] = x0[1]
VMIN[nk+2] = VMAX[nk+2] = x0[2]
VMIN[nk+3] = VMAX[nk+3] = x0[3]
VMIN[nk+4] = VMAX[nk+4] = x0[4]

# Terminal constraint
#VMIN[nv-5] = VMAX[nv-5] = x_target[0]
#VMIN[nv-4] = VMAX[nv-4] = x_target[1]
#VMIN[nv-3] = VMAX[nv-3] = x_target[2]
#VMIN[nv-2] = VMAX[nv-2] = x_target[3]


# Initial solution guess
VINIT = np.zeros(nv)

# Constraint function with bounds
g = [];  g_min = []; g_max = []

# Build up a graph of integrator calls
for k in range(nk):
  # Local state vector
  Xk      = ca.vertcat( (X0[k],X1[k],X2[k],X3[k],X4[k]) )
  Xk_next = ca.vertcat( (X0[k+1],X1[k+1],X2[k+1],X3[k+1],X4[k+1]) )
  
  # Call the integrator
  Xk_end = integrator( {'x0':Xk,'p':U[k] } )['xf']
  
  # append continuity constraints
  g.append(Xk_next - Xk_end)
  g_min.append(np.zeros(Xk.nnz()))
  g_max.append(np.zeros(Xk.nnz()))

# Objective function: L(T)
f = X2[nk]

# Continuity constraints: 0<= x(T(k+1)) - X(T(k)) <=0
g = ca.vertcat(g)

# Create NLP solver instance
opts = {'linear_solver': 'ma27'}
nlp = ca.MXFunction("nlp", ca.nlpIn(x=V), ca.nlpOut(f=f,g=g))
solver = ca.NlpSolver("solver", "ipopt", nlp, opts)

# Solve the problem
sol = solver({"lbx" : VMIN,
              "ubx" : VMAX,
              "x0" : VINIT,
              "lbg" : np.concatenate(g_min),
              "ubg" : np.concatenate(g_max)})

# Retrieve the solution
v_opt = sol["x"]
u_opt  = np.array( v_opt[0:nk] )
x0_opt = np.array( v_opt[nk+0::5] )
x1_opt = np.array( v_opt[nk+1::5] )
x2_opt = np.array( v_opt[nk+2::5] )
x3_opt = np.array( v_opt[nk+3::5] )

# Get values at the beginning of each finite element
tgrid_x = np.linspace( 0, tf, nk+1 )
tgrid_u = np.linspace( 0, tf, nk )

print( 'tgrid_x shape: ' + str( tgrid_x.shape ) )
print( 'x0_opt shape: ' + str( x0_opt.shape ) )

# Plot the results
plt.figure(1)
plt.clf()
plt.subplot( 2, 1, 1 )
plt.plot( tgrid_x, x0_opt, 'b-', label='x0' )
plt.plot( tgrid_x, x1_opt, 'g-', label='x1' )
plt.plot( tgrid_x, x2_opt, 'r-', label='x2' )
plt.plot( tgrid_x, x3_opt, 'b+', label='x3' )

plt.subplot( 2, 1, 2 )
plt.plot( tgrid_u, u_opt, 'b-', label='u' )
plt.title( 'Cartpole - multiple shooting' )
plt.xlabel( 'time' )
plt.legend()
plt.grid()
plt.show()
