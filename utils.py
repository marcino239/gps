import numpy as np

from scipy.stats import multivariate_normal
from sklearn import linear_model, mixture



class MVN( object ):
	"""
		linear algebra on gaussian multivariate
	"""
	
	def __init__( self, mean, cov, A=None ):

		self.mean_ = mean
		self.cov_ = cov

		self.dim_ = mean.shape[ mean.ndim - 1 ]

		if A is None:
			self.A_ = np.eye( self.dim_ )
		else:
			assert( self.dim_ == A.shape[0] )
			self.A_ = A

	def add( self, mvn, c=None ):
		assert( isinstance( mvn, MVN ) )
		assert( self.dim_ == mvn.dim_ )

		if c is None:
			mi = np.dot( self.A_, self.mean_ ) + np.dot( mvn.A_, mvn.mean_ ) 
		else:
			assert( self.dim == c.shape[0] )
			mi = np.dot( self.A_, self.mean_ ) + np.dot( mvn.A_, mvn.mean_ ) + c

		cov = np.dot( np.dot( self.A_, self.cov_ ), self.A_.T ) + \
				np.dot( np.dot( mvn.A_, mvn.cov_ ), mvn.A_.T )

		return MVN( mean, cov )

	def cond( self, a ):
		""" 
			returns conditional p( x1 |x2 = a )
			from N( [x1,x2], cov )
		""" 

		x2dim = a.shape[ a.ndim - 1 ]
		assert( x2dim < self.dim_ )

		x1dim = self.dim_ - x2dim
		mi1 = self.mean_[ :x1dim ]
		mi2 = self.mean_[ x2dim: ]
		cov11 = self.cov_[ :x1dim, :x1dim ]
		cov22 = self.cov_[ x2dim:, x2dim: ]
		cov12 = self.cov_[ :x1dim, x2dim: ]
		cov21 = self.cov_[ x2dim:, :x1dim ]
		
		t = np.dot( cov12, np.linalg.inv( cov22 ) )
		mi = mi1 + np.dot( t, a - mi2 )
		cov = cov11 - np.dot( t, cov21 )

		return MVN( mi, cov )

	def mul( self, mvn ):
		assert( isinstance( mvn, MVN ) )
		assert( self.dim_ == mvn.dim_ )
		
		C = multivariate_normal.pdf( self.mean_, mvn.mean_, self.cov_ + mvn.cov_ )
		
		cov1_1 = np.linalg.inv( self.cov )
		cov2_1 = np.linalg.inv( mvn.cov )
		
		cov = np.linalg.inv( cov1_1 + cov2_1 )
		mi = np.dot( cov, np.dot( cov1_1, self.mi ) + np.dot( cov2_1, mvn.mean ) )
		
		return MVN( mi, cov, C )

		
	def KLDivergence( selvf, mvn ):
		"""
			calculates KL( self || mvn )
		"""
		assert( isinstance( mvn, MVN ) )
		assert( self.dim_ == mvn.dim_ )
		
		m = mvn.mean_ - self.mean_
		c_1 = np.linalg.inv( mvn.cov_ )
		
		kl = np.trace( c_1.dot( self.cov_ ) ) + m.T.dot( c_1 ).dot( m ) \
				- self.dim_ + np.log( np.linalg.det( mvn.cov_ ) / np.linalg.det( seld.cov_ ) )

		return kl / 2.0
		
	def rand( self, size=1 ):
		#TODO - implement mvn random generator for a given size
		return None
		
	def log_params( self ):
		" returns constant and matrix "
		c, A = np.log( np.linalg.det( self.cov_ ) ) - self.dim_ * np.log( 2. * np.pi ), np.linalg.inv( self.cov_ )
		return 0.5 * c, -0.5 * A, self.mean_

	def log( self, x ):
		# returns log of probability distribution
		d = x - self.mean_
		c, A,  = self.log_params()

		l = c + d.T.dot( A ).dot( d )
		return l


	
class Lin_Gaussian_Estimator( object ):
	def __init__( self, x_len ):
		"""
			Estimates linear gaussian dynamics in the form:
			P( xt+1 | xt, ut ) = N( fxt xt + fut ut , Ft )
		"""
		pass


	def fit( self, X_dynamics, var_ratio = 0.1, min_var = 0.1, num_gaussians=1 ):
		"""
			fits linear gaussian regulator in the form of:
			p( xt+1 | xt, ut ) = N( fxt * xt + fut * ut, Ft )
			
			X_dynamics = [rollout 0] [ x(t+1,0), x(t,0), u(t,0) ]
							...
						 [rollout 0] [ x(t+1,k), x(t,k), u(t,k) ]
			x(t+1,k), x(t,k) are x_len long
		"""

		assert( X_dynamics.ndims == 3 )
		assert( X_dynamics.shape[ 2 ] > x_len * 2 )

		lr = linear_model.LinearRegression( fit_intercept = False )
		lr.fit( X_dynamics[ :, self.x_len: ], X_dynamics[ :, :self.x_len ] )
		
		self.fx = lr.coef_
		self.fu = lr.intercept_
				
		if X_dynamics.shape[0] == 1:
			self.F = np.diag( np.max( np.row_stack( ( X_dynamics[ 0, :x_len ] * var_ratio,
														np.ones( x_len ) * min_var ) ), axis=0 ) )
		else:
			# use GMM to estimate covariance
			gmm = mixture.GMM( n_components=1, covariance_model='full' )
			gmm.fit( X_dynamics )
			
			self.F = gmm.covar_[ 0, : ]

class Lin_Gaussian_Controller( object ):
	"""
		implements linear gaussian controller in the form of:
		
		P( ut | xt ) = P( ut; u_hat + kt + Kt( xt - x_hat ), F )
	"""

	def __init__( self, u_hat, x_hat, k, K, F ):
		self.k = k
		self.K = k
		self.F = F
		
		self.mvn = MVN( u_hat + k - K.dot( x_hat ), F )
	
	def controll( self, x ):
		return self.mvn.rand() + self.K.dot( x )
