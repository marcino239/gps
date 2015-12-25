import numpy as np

from utils import *
from ilqr import LQR

# implements guided policy search algorithm
# based on work of S Levine, P Abbeel
#

class GPS( object ):
	def __init__( self, params ):
		self.x_len = params[ 'x_len' ]
		self.u_len = params[ 'u_len' ]
		self.num_gaussians = params[ 'num_gaussians' ]

	def estimate_linear_controllers( self, xu_train ):
		# estimate linear gaussian controller in the form of x+1 = x + kx + noise

		estimated_linear_controllers = []

		# create X_dynamics with organisation:
		#   [num rollouts] [time step] [xt+1 xt u]
		XU_dynamics = np.concatenate( (xu_train[ :, 1:, :self.x_len], xu_train[ :, :-1, :]), axis=2 )

		# swap rollout and time axe
		xu = np.swapaxes( XU_dynamics, 0, 1 )
				
		# for each timestep
		for i in range( xu.shape[ 0 ] ):
			lin_reg = Lin_Gaussian_Estimator( self.x_len )
			lin_reg.fit( xu[i, :, :], self.x_len, num_gaussians=self.num_gaussians )
			estimated_linear_controllers.append( lin_reg )

		# copy last controller to ensure we can span whole time range
		estimated_linear_controllers.append( lin_reg )

		return estimated_linear_controllers


	def train( self, xu_train, o_train, system_cost, gps_params ):
		"""
			x_train - training state data
			o_train - training image data
			objective_f - objective function

			x_train dimensions:  [ num_rollouts ][ num_time steps ][ x_vec, u_vec ]
			o_train dimensions:  [ num_rollouts ][ num_time steps ][ width, height ]

			returns Policy
		"""
#		assert( x_train.shape[0] == o_train.shape[0] )
#		assert( x_train.shape[1] == o_train.shape[1] )
# TODO - better error handling

		# TODO - add diagnostics and info capture

		# estimate linear controllers
		estimated_linear_controllers = self.estimate_linear_controllers( xu_train )

		lqr = LQR( self.x_len, self.u_len )

		# TODO run loop for K times or when error improvement is less than epsilon
		for k in range( gps_params[ 'K'] ):
			# optimise supervised learning
			# TODO:
			#   sample x, o from stored states and their coresponding observations
			#   minimise objective function
			#   use SGD to train
			
			# Questions:
			#  how is covariance Cti estimated = this is covariance of p( ut | xt )
			#  how is importance sampling utilized?
			#  how is training data prepared?
			

			# optimise linear controllers using LQR
			linear_controllers = lqr.LQR( xu_train, estimated_linear_controllers, system_cost, 1.0 )		# TODO add lagrange multipliers

			# update Lagrange multipliers

		return policy
		
	def merge_data( x_train, o_train, x_run, o_run ):
		pass
