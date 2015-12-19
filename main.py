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
	
	x_len = C.get_x_len()
	u_len = C.get_u_len()
	gps_params[ 'x_len' ] = x_len
	gps_params[ 'u_len' ] = u_len

	# load training data
	# training data contains state and image data
	# TODO: modify to deal with number of rollouts
	
	f = h5py.File( 'data.h5', 'r' )
	xu_train_orig = f[ 'x' ].value
	
#	if 'o' not in f:
#		print( 'WARNING: observations not in data' )
#		o_train_orig = None
#	else:
#		o_train_orig = f[ 'o' ]
#		o_train_orig = o_train_orig[ ::gps_params[ 'resample' ]	]
	o_train = None

	if xu_train_orig.ndim == 2:
		xu_train_orig = xu_train_orig[ :, 1: ]	# TODO - at the moment 1st column is time

		s = xu_train_orig.shape
		xu_train_orig = xu_train_orig.reshape( (1,s[0], s[1]) )

	resample_idx = range( 0, xu_train_orig.shape[1], gps_params[ 'resample' ] )

	xu_train_orig = np.take( xu_train_orig, resample_idx, axis=1 )
	xu_train = xu_train_orig
	x_train = xu_train_orig[ :,:, :x_len ]
	u_train = xu_train_orig[ :,:, x_len ]			# TODO check table organisation

	gps = GPS( gps_params )
	training_errors = []
		
	# loop untill convergence:
	for k in range( gps_params[ 'K'] ):
		print( 'running for k: {0}'.format( k ) )

		# execute gps
		policy, training_error = gps.train( xu_train, o_train, C.get_system_cost(), gps_params )
		training_errors.append( training_error )

		# run on controller and collect the data
		xu_run = C.run( policy )

		# merge data
		xu_train = gps.merge_data( xu_train, o_train, xu_run, o_run )

	# display training error


if __name__ == "__main__":

	if len( sys.argv ) < 3:
		print( 'usuage: gps.py cartpole training_data.h5 policy_file.pkl' )
		sys.exit()

	main( sys.argv[1], sys.argv[2], sys.argv[3] )
