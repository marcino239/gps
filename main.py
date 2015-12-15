import numpy as np
import sys
import h5py
import utils

import controller
from gps import GPS

# TODO:
#   estimate linear controllers
#   optimise linear controllers using LQR
#   optimise supervised learning
#   update Lagrange multipliers
#   repeat


gps_params_default = { 
						'resample':			6,
						'num_gaussians': 	1,
						'K':				1,
						'epsilon':			1e-5,
					}


def main( controller_type, training_data, policy_file, gps_params = gps_params_default ):
	
	# create controller
	if controller_type == 'cartpole':
		C = controller.CartPole()
		C.set_system_cost( x0=np.array( [ 0, 0, 0, 0 ] ), u0=0.0, Wx=np.eye( 4 ), Wu=1e-3 )
	else:
		print( 'not implemented: ' + controller_type )
	
	gps_params[ 'x_len' ] = C.get_x_len()
	gps_params[ 'u_len' ] = C.get_u_len()

	# load training data
	# training data contains state and image data
	f = h5py.File( 'data.h5', 'r' )
	x_train_orig = f[ 'x' ]
	o_train_orig = f[ 'o' ]						# TODO: modify to deal with number of rollouts
	x_train_orig = x_train_orig[ ::gps_params[ 'resample' ]	]
	o_train_orig = o_train_orig[ ::gps_params[ 'resample' ]	]

	x_train = x_train_orig.copy()
	o_train = o_train_orig.copy()

	gps = GPS( gps_params )
	training_errors = []
	
	# loop untill convergence:
	for k in range( gps_params[ 'K'] )
		print( 'running for k: {0}'.format( k ) )

		# execute gps
		policy, training_error = gps.train( x_train, o_train, C.get_system_cost(), gps_params )
		training_errors.append( training_error )

		# run on controller and collect the data
		x_run, u_run = C.run( policy )

		# merge data
		x_train, o_train = gps.merge_data( x_train, o_train, x_run, o_run )

	# display training error


if __name__ == "__main__":

	import ipdb
	ipdb.set_trace()
	
	if len( sys.argv ) < 3:
		print( 'usuage: gps.py cartpole training_data.h5 policy_file.pkl' )
		sys.exit()

	main( sys.argv[1], sys.argv[2], sys.argv[3] )
