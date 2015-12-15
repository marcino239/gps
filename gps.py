import numpy as np

from utils import *

# implements guided policy search algorithm
# based on work of S Levine, P Abbeel
#

class GPS( object )
	def __init__( self, **kwargs )
		self.x_len = kwargs[ 'x_len' ]
		self.u_len = kwargs[ 'u_len' ]

	def estimate_linear_controllers( self, x_train, num_gaussians ):
		# estimate linear gaussian controller in the form of x+1 = x + kx + noise

		estimated_linear_controllers = []
		
		# swap rollout and time axe
		x = np.swapaxes( x_train, 0, 1 )
		
		# for each timestep
		for i in range( x.shape[ 0 ] ):
			lin_reg = Lin_Gaussian_Estimator()
			lin_reg.fit( x[i, :, :], self.x_len, num_gaussians=num_gaussians )
			estimated_linear_controllers.append( lin_reg )

		return estimated_linear_controllers


	def optimise_linear_controllers( self, estimated_linear_controllers, objective_f ):
		lqr = LQR( self.x_len )
		
		linear_controllers = lqr.LQR( trajectory, estimated_linear_controllers, system_cost ):
		

	def train( x_train, o_train, system_cost, gps_params ):
		"""
			x_train - training state data
			o_train - training image data
			objective_f - objective function

			x_train dimensions:  [ num_rollouts ][ num_time steps ][ x_vec, u_vec ]
			o_train dimensions:  [ num_rollouts ][ num_time steps ][ width, height ]

			returns Policy
		"""
		assert( x_train.shape[0] == o_train.shape[0] )
		assert( x_train.shape[1] == o_train.shape[1] )

		# TODO - add diagnostics and info capture

		# optimise supervised learning

		# estimate linear controllers
		linear_controllers = estimate_linear_controllers( x_train, kwargs[ 'num_gaussians' ] )

		# optimise linear controllers using LQR

		# update Lagrange multipliers


		return policy
		
