import numpy as np
import numdifftools as ndt

from utils import Lin_Gaussian_Controller

class LQR( object ):
	def __init__( self, x_len, u_len ):
		self.x_len = x_len
		self.u_len = u_len


	def LQR( self, xu_vec, lin_regs_estimated, system_cost, ni ):
		"""
			trajectory - vector of (xi,ui)  dimensions: steps, x_u 
			lin_regs - linear gaussian regulators for all T time steps
			system_cost = [ l(x,u,t), lxut, lxu_xut ]

			returns: estimated linear controllers

			LQR procedure:

			Q xu,xut = l xu,xut + f xut V x,xt+1 f xut
			Q xut = xut + f xut V xt+1
			V x,xt = Q x,xt - Q Tu,xt Q u,ut Q u,xt
			V xt = Q xt - Q Tu,xt Q u,ut Q ut

			for t = 0..N-1

			Notes:
			VxN = dlN(x,u) / dx				# Jacobian
			Vx,xN = d^2 lN(x,u) / dx^2		# Hessian
		"""

		get_x   = lambda xu: xu[ :self.x_len ]
		get_u   = lambda xu: xu[ self.x_len: ]
		get_x_x = lambda xu: xu[ :self.x_len, :self.x_len ]
		get_u_u = lambda xu: xu[ self.x_len:, self.x_len: ]
		get_u_x = lambda xu: xu[ self.x_len:, :self.x_len ]			# TODO - check this, need to return x
		get_xu  = lambda x, u: np.concatenate( [x,u] )
		get_ux  = lambda x, u: np.concatenate( [u,x] )


		# TODO - where is this in the loop
		def l_star( system_cost, ni, lin_regs_estimated ):

			# new augumented cost.  Depends on segement!
			#  c*( t ) = c( t ) / ni - log( prior( linear_controller( t ) )

			l_      = system_cost[ 'l' ]
			lx_     = system_cost[ 'lx' ]
			lxu_    = system_cost[ 'lxu' ]
			lx_x_   = system_cost[ 'lx_x' ]
			lxu_xu_ = system_cost[ 'lxu_xu' ]

			c, P, K, x_mean = lin_regs_estimated.log_params()

			x_mean_xu = np.concatenate( (x_mean, np.zeros( self.u_len )) )

			xu_len = self.x_len + self.u_len
			K_xu = np.zeros( (xu_len,1) )
			K_xu[ :self.x_len, :1 ] = K

			P_xu = np.zeros( (xu_len, xu_len) )
			P_xu[ :self.x_len, :self.x_len ] = P

			P_T_P_xu_J = (P_xu.T + P_xu).dot( K_xu ).T
			P_T_P_xu_H = (P_xu.T + P_xu).dot( K_xu ).dot( K_xu.T ).T
			P_T_P_J = (P.T + P).dot( K ).T
			P_T_P_H = (P.T + P).dot( K ).dot( K.T ).T

			inv_ni = 1.0 / ni

			def l( xu ):
				d = K_xu.T.dot( x ) - x_mean_xu
				return l_( xu ) * inv_ni - c - d.T.dot( P_xu ).dot( d )

			def lxu( xu ):
				d = K_xu.T.dot( xu ) - x_mean_xu
				return lxu_( xu ) * inv_ni - P_T_P_xu_J.dot( d.T )

			def lxu_xu( xu ):
				return lxu_xu_( xu ) * inv_ni - P_T_P_xu_H
			
			def lx( x ):
				d = K.T.dot( x ) - x_mean
				return lx_( x ) / ni - P_T_P_J.dot( d )
				
			def lx_x( x ):
				d = x - x_mean
				return lx_x_( x ) / ni - P_T_P_H

			return l, lx, lx_x, lxu, lxu_xu

		T = xu_vec.shape[1] - 1
		l, lx, lx_x, lxu, lxu_xu = l_star( system_cost, ni, lin_regs_estimated[ T ] )

		Vx     = [ None ] * (T+2)
		Vx_x   = [ None ] * (T+2)
		Qxu_xu = [ None ] * (T+1)
		Qxu    = [ None ] * (T+1)
		Qx_x   = [ None ] * (T+1)
		Qu_u_i = [ None ] * (T+1)
		Qu_x   = [ None ] * (T+1)
		k      = [ None ] * (T+1)
		K      = [ None ] * (T+1)
		lin_controllers = [ None ] * (T+1)

		# initialise gradients of V
		x_terminal = get_x( xu_vec[ -1, T, : ] )
		Vx[ T+1 ] =  lx( x_terminal )
		Vx_x[ T+1 ] =  lx_x( x_terminal )

		# for t = T..0
		for t in range( T, -1, -1 ):
			
			linreg_t = lin_regs_estimated[ t ]
			l, lx, lx_x, lxu, lxu_xu = l_star( system_cost, ni, linreg_t )
			
			fx_t = linreg_t.fx
			fu_t = linreg_t.fu
			fxu_t = linreg_t.fxu

			xu_t = xu_vec[ -1, t, : ]

			lxu_xut = lxu_xu( xu_t )

			#	Q_xu,xut = l_xu,xut + f_xut^T V_x,xt+1 f_xut
			Qxu_xu[t] = lxu_xut + fxu_t.T.dot( Vx_x[t+1] ).dot( fxu_t )
			Qu_u_i[t] = np.linalg.inv( get_u_u( Qxu_xu[t] ) )
			Qx_x[t]   = get_x_x( Qxu_xu[t] )
			Qu_x[t]   = get_u_x( Qxu_xu[t] )

			#	Q xut = l_xut + f_xut V_xt+1
			Qxu[t] = lxu( xu_t ) + fxu_t.T.dot( Vx[t+1] )

			#	V_x,xt = Q_x,xt - Q_u,xt^T Q_u,ut^-1 Q_u,xt
			Vx_x[t] = Qx_x[t] - Qu_x[t].T.dot( Qu_u_i[t] ).dot( Qu_x[t] )

			#	V_xt = Q_xt - Qu,xt^T Q_u,ut^-1 Q ut
			Vx[t] = get_x( Qxu[t] ) - Qu_x[t].T.dot( Qu_u_i[t] ).dot( get_u( Qxu[t] ) )
			
			#	k_t = -Q_uut^-1 Q_ut
			k[t] = -Qu_u_i[t].dot( get_u( Qxu[t] ) )
			
			#	k_t = -Q_uut^-1 Q_uxt
			K[t] = -Qu_u_i[t].dot( Qu_x[t] )

			# TODO check range

			#  u_hat, x_hat are states of current trajectory
			x_hat = xu_vec[ -1, t, :self.x_len ]
			u_hat = xu_vec[ -1, t, self.x_len: ]

			# return linear controllers
			lin_controllers[t] = Lin_Gaussian_Controller( u_hat, x_hat, k[t], K[t], -K[t] )

		return lin_controllers
