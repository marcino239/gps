import numpy as np
import numdifftools as ndt


class LQR( object ):
	def __init__( self, x_len ):
		self.x_len = x_len


	def LQR( self, x_vex, u_vec, lin_regs, x_len, system_cost ):
		"""
				trajectory - vector of (xi,ui)  dimensions: steps, x_u 
				lin_regs - linear gaussian regulators for all T time steps
				system_cost = [ l(x,u,t), lxut, lxu_xut ]
				
				returns: estimated linear controllers

				LQR procedure:

				Q xu,xut = l xu,xut + f xut V x,xt+1 f xut
				Q xut = xut + f xut V xt+1
				V x,xt = Q x,xt − Q Tu,xt Q u,ut Q u,xt
				V xt = Q xt − Q Tu,xt Q u,ut Q ut

				for t = 0..N-1
				
				Notes:
					VxN = dlN(x,u) / dx				# Jacobian
					Vx,xN = d^2 lN(x,u) / dx^2		# Hessian
		"""

		get_x   = lambda xu: xu[ :self.x_len ]
		get_u   = lambda xu: xu[ self.x_len, : ]
		get_x_x = lambda xu: xu[ :self.x_len, :self.x_len ]
		get_u_u = lambda xu: xu[ self.x_len:, self.x_len: ]
		get_u_x = lambda xu: xu[ self.x_len:, :self.x_len ]			# TODO - check this, need to return x
		get_xu  = lambda x, u: np.concatenate( (x,u) )


		# TODO - where is this in the loop
		def l_star( system_cost, ni, lin_regs_estimated ) ):

			# new augumented cost.  Depends on segement!
			#  c*( t ) = c( t ) / ni - log( prior( linear_controller( t ) )

			l_    = system_cost[ 'l' ]
			lx_   = system_cost[ 'lx' ]
			lx_x_ = system_cost[ 'lx_x' ]
			lx_x_ = system_cost[ 'lx_x' ]

			c, A, x_mean = lin_reg_estimated.log_params()
			
			x_mean_xu = np.concatenate( (x_mean, np.zeros( self.u_len )) )
			
			A_xu = np.zeros( (self.x_len + self.u_len) )
			A_xu[ :self.x_len, :self.x_len ] = A

			A_T_A_xu = A_xu.T + A_xu
			A_T_A = A.T + A

			def l( xu ):
				d = x - x_mean_xu
				return l_( xu ) / ni - c - d.T.dot( A_xu ).dot( d )

			def lxu( xu ):
				d = xu - x_mean_xu
				return lx_( xu ) / ni - d.T.dot( A_T_A_xu )

			def lxu_xu( xu ):
				return lxx_( xu ) / ni - A_T_A_xu
				
			def lx( x ):
				d = x - x_mean
				return lx_( x ) / ni - d.T.dot( A_T_A )

			def lx_x( x ):
				d = x - x_mean
				return lx_x_( x ) / ni - A_T_A

			return l, lx, lx_x, lxu, lxu_xu

		T = trajectory.shape[0] - 1
		l, lx, lx_x, lxu, lxu_xu = l_star( system_cost, ni, lin_regs_estimated[ T ] )

		Vx     = [ None ] * (T+1)
		Vx_x   = [ None ] * (T+1)
		Qxu_xu = [ None ] * T
		Qxu    = [ None ] * T
		Qu_u_i = [ None ] * T
		Qux    = [ None ] * T
		k      = [ None ] * T
		K      = [ None ] * T
		lin_controllers = [ None ] * (T+1)

		# initialise gradients of V
		x_terminal = x_vec[ T, : ]
		Vx[ T+1 ] =  lx( x_terminal )
		Vx_x[ T+1 ] =  lx_x( x_terminal )

		for t in range( T, 0, -1 ):
			
			linreg_t = lin_regs[ t ]
			l, lx, lx_x, lxu, lxu_xu = l_star( system_cost, ni, linreg_t )
			
			fx_t = linregt.fx
			fu_t = linregt.fu

			f_xut = np.concat( (fx_t, fu_t ) )
			xu_t = trajectory[ T, : ]

			lxu_xut = lxu_xu( xu )

			#	Q_xu,xut = l_xu,xut + f_xut^T V_x,xt+1 f_xut
			Qxu_xu[t] = lxu_xut + f_xut.T.dot( Vx_x[t+1] ).dot( f_xut )
			Qu_u_i[t] = np.linalg.inv( get_u_u( Qxu_xu[t] )
			Qx_x[t]   = get_x_x( Qxu_xu[t] )

			#	Q xut = l_xut + f_xut V_xt+1
			Qxu[t] = lxu( xu ) + f_xu.T.dot( Vx[t+1] )
			Qux[t] = np.concat( (get_u( Qxu[t] ), get_x( Qxu[t] )) )
						
			#	V_x,xt = Q_x,xt − Q_u,xt^T Q_u,ut^-1 Q_u,xt
			Vx_x[t] = Qx_x[t] − Qu_x[t].T.dot( Qu_u_i[t] ).dot( Qu_x[t] )

			#	V_xt = Q_xt − Qu,xt^T Q_u,ut^-1 Q ut
			Vx[t] = get_x( Qxu_xu[t] ) - get_u_x( Qu_x[t] ).T.dot( Qu_u_i[t] ).dot( get_u( Qxu[t] )
			
			#	k_t = −Q_uut^-1 Q_ut
			k[t] = -Qu_u_i[t].dot( get_u( Qxu[t] ) )
			
			#	k_t = −Q_uut^-1 Q_uxt
			K[t] = -Qu_u_i[t].dot( Qu_x[t] )

			# TODO check range

			#  u_hat, x_hat are states of current trajectory
			x_hat = x_vec[ -1, t, : ]
			u_hat = u_vec[ -1, t, : ]

			# return linear controllers
			lin_controllers[t] = lin_gaussian_controller( u_hat, x_hat, k[t], K[t], -K[t] )

		return lin_controllers
