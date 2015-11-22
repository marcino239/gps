import numpy as np
from scipy.stats import multivariate_normal

class MVN( object ):
	def __init__( self, mean, cov, A=None ):

		self.mean = mean
		self.cov = cov

		self.dim = mean.shape[ mean.ndim - 1 ]

		if A is None:
			self.A = np.eye( self.dim )
		else:
			assert( self.dim == A.shape[0] )
			self.A = A

	def add( self, mvn, c=None ):
		assert( isinstance( mvn, MVN ) )
		assert( self.dim == mvn.dim )

		if c is None:
			mi = np.dot( self.A, self.mean ) + np.dot( mvn.A, mvn.mean ) 
		else:
			assert( self.dim == c.shape[0] )
			mi = np.dot( self.A, self.mean ) + np.dot( mvn.A, mvn.mean ) + c

		cov = np.dot( np.dot( self.A, self.cov ), self.A.T ) + \
				np.dot( np.dot( mvn.A, mvn.cov ), mvn.A.T )

		return MVN( mean, cov )

	def cond( self, x2, a ):
		""" 
			returns conditional p(x 1 |x 2 = a)
			from N( x1;x2, cov )
		""" 

		x2dim = x2.shape[0]
		x1dim = self.dims - x2dim
		mi1 = self.mean[ :x1dim ]
		mi2 = self.mean[ x2dim: ]
		cov11 = self.cov[ :x1dim, :x1dim ]
		cov22 = self.cov[ x2dim:, x2dim: ]
		cov12 = self.cov[ :x1dim, x2dim: ]
		cov21 = self.cov[ x2dim:, :x1dim ]
		
		t = np.dot( cov12, np.linalg.inv( cov22 ) )
		mi = mi1 + np.dot( t, a - mi2 )
		cov = cov11 - np.dot( t, cov21 )

		return MVN( mi, cov )

	def mul( self, mvn ):
		assert( isinstance( mvn, MVN ) )
		assert( self.dim == mvn.dim )
		
		C = multivariate_normal.pdf( self.mean, mvn.mean, self.cov + mvn.cov )
		
		cov1_1 = np.linalg.inv( self.cov )
		cov2_1 = np.linalg.inv( mvn.cov )
		
		cov = np.linalg.inv( cov1_1 + cov2_1 )
		mi = np.dot( cov, np.dot( cov1_1, self.mi ) + np.dot( cov2_1, mvn.mean ) )
		
		return MVN( mi, cov, C )
