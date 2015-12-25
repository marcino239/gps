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

nk = 100      # Control discretization
tf = 10.0    # End time
coll = False # Use collocation integrator

# initial and target points
x0       = ca.DMatrix( [0, np.pi/10, 0, 0 ] )
x_target = ca.DMatrix( [0, np.pi  , 0, 0 ] )

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
x  = ca.SX.sym( 'x' , 4 )  # state

# ODE rhs function
#   x[0] = dtheta
#   x[1] = theta
#   x[2] = dx
#   x[3] = x

sint = ca.sin( x[1] )
cost = ca.cos( x[1] )

fric = mic * ca.sign( x[2] )
num  = g * sint + cost * ( -u - mp * l * x[0] * x[0] * sint + fric ) / ( mc + mp ) - mip * x[0] / ( mp * l )
denom = l * ( 4. / 3. - mp * cost*cost / (mc + mp) )
ddtheta = num / denom
ddx = (-u + mp * l * ( x[0] * x[0] * sint - ddtheta * cost ) - fric) / (mc + mp)

x_err = x-x_target
cost_mat = np.diag( [1,1,1,1] )

ode = ca.vertcat([	ddtheta, \
					x[0], \
					ddx, \
					x[2]
				] )

#					ca.mul( [ x_err.T, cost_mat, x_err ] )

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
  opts['abstol'] = 1e-1 # tolerance
  opts['reltol'] = 1e-1 # tolerance
#  opts['steps_per_checkpoint'] = 1000
  opts['quad_err_con'] = True
  opts['fsens_err_con' ] = True 
  opts['t0'] = 0.
  opts['tf'] = tf
  integrator = ca.Integrator('integrator', 'cvodes', dae, opts)

integrator.setInput( x0, 'x0' )
integrator.setInput( 0, 'p' )
integrator.evaluate()
integrator.reset()

ts = np.linspace( 0, tf, nk )

def out( t ):
	integrator.integrate( t )
	return integrator.getOutput().full()

sol = np.array( [out(t) for t in ts] ).squeeze()

# Plot the results
plt.figure(1)
plt.clf()
plt.plot( ts, sol[:,0], 'b-', label='x0' )
plt.plot( ts, sol[:,1], 'g-', label='x1' )
plt.plot( ts, sol[:,2], 'r-', label='x2' )
plt.plot( ts, sol[:,3], 'b+', label='x3' )

plt.xlabel( 'time' )
plt.legend()
plt.grid()
plt.show()
