import numpy as np
import numdifftools as ndt


class CartPole( object ):

	g = 9.81
	mc = 1.0
	mp = 1.
	l = 1.
	mip = 1.
	mic = 5.
	dt = 0.002

	OVERSAMPLE = int( 1. / 0.002 / 60. )
	
	MAX_RECORDS = int( 10.0 * 60. )

	def __init__( self ):
		
		self.minx = -2.
		self.maxx = 2.
		
		self.xout = np.zeros( (4,1) )
		self.state = np.array( [ np.pi, 0., 0, 0.] )
		
		self.records = np.zeros( ( CartPole.MAX_RECORDS, 4 + 1 + 1 ) )
		self.record_counter = 0
		
		self.recording_on = False
		self.start_time = 0.

	def reset( self ):
		# resets back to (pi, pi)
		self.state = np.array( [np.pi, 0., np.pi, 0.] )
				
	def step( self, u ):
		for i in range( CartPole.OVERSAMPLE ):
			self.dynamics( self.state, u )
		
		if self.recording_on:
			if self.record_counter < CartPole.MAX_RECORDS:
				self.records[ self.record_counter, 0 ] = time.clock() - self.start_time
				self.records[ self.record_counter, 1:5 ] = self.state
				self.records[ self.record_counter, 5 ] = u
				self.record_counter += 1

	def update( self, font, screen ):
		# get screen center
		width, height = screen.get_size()
		x0i, y0i = width / 2, height / 2
		
		# get position of pendulum
		x1 = np.sin( self.state[0] ) * CartPole.l + self.state[2]
		y1 = (-np.cos( self.state[0] ) * CartPole.l + 0. ) * float( width ) / float( height )

		x2 = self.state[2]
		y2 = 0.
		
		x3 = x2
		y3 = 0.
		
		x4 = x2 + 0.5
		y4 = y2 - 0.5

		x1i = x0i + int( x1 * width * 0.1 )
		y1i = y0i + int( y1 * height * 0.1 )

		x2i = x0i + int( x2 * width * 0.1 )
		y2i = y0i + int( y2 * height * 0.1 )

		x3i = x0i + int( x3 * width * 0.1 )
		y3i = y0i + int( y3 * height * 0.1 )

		x4i = x0i + int( x4 * width * 0.1 )
		y4i = y0i + int( y4 * height * 0.1 )


		# draw circles
		pg.draw.circle( screen, (255, 255, 255), (x1i, y1i), 5, 0 ) 

		# draw box
		pg.draw.rect( screen, (255, 255, 255), (x3i, y3i, x3i - x4i, y3i - y4i), 2 )

		# draw line
		pg.draw.line( screen, (255, 255, 255), (x1i, y1i), (x2i, y2i), 2 )

		# show recording time
		if self.recording_on:
			r = font.render( "R", True, pg.Color( 'Yellow' ) )
			screen.blit( r, (0, 0) )

	def start_recording( self ):
		self.recording_on = False
		self.record_counter = 0
		self.start_time = time.clock()
		self.recording_on = True

	def dynamics( self, x, F ):
		"""
			x0 - theta
			x1 - d theta / dt
			x2 - x
			x3 - dx / dt
		"""
		
		sint = np.sin( x[0] )
		cost = np.cos( x[0] )
		fric = CartPole.mic * np.sign( x[3] )
		
		num = CartPole.g * sint + cost * ( -F - CartPole.mp * CartPole.l * x[1]**2 * sint + fric ) / ( CartPole.mc + CartPole.mp ) - CartPole.mip * x[1] / ( CartPole.mp * CartPole.l )
		denom = CartPole.l * ( 4. / 3. - CartPole.mp * cost**2 / (CartPole.mc + CartPole.mp) )
		theta_acc = num / denom
		x[1] = x[1] + CartPole.dt * theta_acc
		x[0] = x[0] + CartPole.dt * x[1]
		
		x_acc = (F + CartPole.mp * CartPole.l * ( x[1]**2 * sint - theta_acc * cost ) - fric) / (CartPole.mc + CartPole.mp)
		x[3] = min( max( x[3] + CartPole.dt * x_acc, self.minx ), self.maxx )
		x[2] = x[2] + CartPole.dt * x[3]


	def save_data( self, fout_name ):
		"""
			X = [ time, theta, dtheta, x, dx, u ]
		"""

		f = h5py.File( fout_name,  'w' )
		f.create_dataset( 'x', data=self.records )
		f.create_dataset( 'x_len', data=self.record_counter )

	@staticmethod
	def get_l():
		"""
			calculates cost function, Jacobian and Hessian
			l, lxu, lxu_xu
			
			these functions take the following parameters:
			xu, wx, wu, x0
		"""

		def l( xu, wx, wu, x0 ):
			x_ = xu[ 0:4] - x0
			u_ = xu[ 4: ]
			return x_.T.dot( wx ).dot( x_ ) + u_ * wu * u_

		return l, ndt.Jacobian( l ), ndt.Hessian( l )


	@staticmethod
	def get_x_len():
			return 4

	@staticmethod
	def get_u_len():
			return 1

	@staticmethod
	def get_dynamics( X ):
		"""
			dynamics: [ xt+1, xt, ut ]
		"""
		return np.column_stack( ( X[ 1:, 1:5 ], X[ :-1, 1: ] ) )

	@staticmethod
	def get_x( X ):
		return x[ :, 1:5 ]
	
	@staticmethod
	def get_u( X ):
		return x[ :, 5 ] 

	def set_system_cost( self, x0, u0, Wx, Wu ):
		self.cost_params = { 'x0': x0, 'u0': u0, 'Wx': Wx, 'Wu': Wu }

	def get_system_cost( self ):
		if self.cost_params is None:
			raise ValueError( 'cost_params are none' )

		x0  = self.cost_params[ 'x0' ]
		u0  = self.cost_params[ 'u0' ]
		Wu  = self.cost_params[ 'Wx' ]
		Wx  = self.cost_params[ 'Wu' ]

		x_len = self.get_x_len()
		u_len = self.get_u_len()
		Wxu = np.zeros( (x_len + u_len, x_len + u_len ) )
		Wxu[ :x_len, :x_len ] = Wx
		Wxu[ x_len:, x_len: ] = Wu

		xu0 = np.conctatenate( (x0, u0) )
		A_T_A_xu = Wxu.T + Wxu
		A_T_A_x  = Wx.T + Wx

		# jacobian of quadratic form: l = x^T A x
		# dl/dx = x^T (A^T + A)
		#
		# hessian of quadratic form
		# d^2 l / dl^2 = A^T + A

		def l( xu ):
			# x is concatenated vector of x and u
			d = xu - xu0
			return d.T.dot( Wxu ).dot( d )

		def lx( x ):
			d = x - x0
			return d.T.dot( A_T_A_x )

		def lxu( xu ):
			d = xu - xu0
			return d.T.dot( A_T_A_xu )

		def lx_x( x ):
			return A_T_A_x
			
		def lxu_xu( xu )
			return A_T_A_xu

		return { 'l':l, 'lx': lx, 'lx_x': lx_x, 'lxu': lxu, 'lxu_xu': lxu_xu }
