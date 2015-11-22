import pygame as pg
import numpy as np
import sys
import time
import h5py

USE_FORCE = 100.

class DoublePendulum( object ):
	M1 = 1.
	M2 = 1.
	L1 = 1.
	L2 = 1.
	G = 9.8
	mu = 0.1
	dt = 0.002

	OVERSAMPLE = int( 1. / 0.002 / 60. )

	def __init__( self, minu, maxu ):
		
		self.minu = minu
		self.maxu = maxu
		
		self.xout = np.zeros( (4,1) )
		
		self.state = np.array( [0, 0., 0, 0.] )
		
	def reset( self ):
		# resets back to (pi, pi)
		self.state = np.array( [np.pi, 0., np.pi, 0.] )
				
	def step( self, u ):
		for i in range( DoublePendulum.OVERSAMPLE ):
			self.dynamics( self.state, u, self.xout )

	def update( self, screen ):
		# get screen center
		width, height = screen.get_size()
		x0i, y0i = width / 2, height / 2
		
		# get position of pendulum
		x1 = np.sin( self.state[0] ) * DoublePendulum.L1
		y1 = np.cos( self.state[0] ) * DoublePendulum.L1

		x2 = np.sin( self.state[2] ) * DoublePendulum.L2 + x1
		y2 = np.cos( self.state[2] ) * DoublePendulum.L2 + y1

		x1i = x0i + int( x1 * width * 0.2 )
		y1i = y0i + int( y1 * height * 0.2 )

		x2i = x0i + int( x2 * width * 0.2 )
		y2i = y0i + int( y2 * height * 0.2 )

		# draw center circles
		pg.draw.circle( screen, (255, 255, 255), (x0i, y0i), 10, 0 ) 
		pg.draw.circle( screen, (255, 255, 255), (x1i, y1i), 10, 0 ) 
		pg.draw.circle( screen, (255, 255, 255), (x2i, y2i), 10, 0 ) 

		# draw lines
		pg.draw.line( screen, (255, 255, 255), (x0i, y0i), (x1i, y1i), 2 )
		pg.draw.line( screen, (255, 255, 255), (x1i, y1i), (x2i, y2i), 2 )


	def dynamics( self, x, u, dydx ):

		dydx[0] = x[1]

		del_ = x[2]-x[0]
		den1 = (DoublePendulum.M1+DoublePendulum.M2)*DoublePendulum.L1 - \
				DoublePendulum.M2*DoublePendulum.L1*np.cos(del_)*np.cos(del_)
		dydx[1] = ( DoublePendulum.M2*self.L1 * x[1] * x[1] * np.sin(del_) * np.cos(del_)
			   + DoublePendulum.M2*DoublePendulum.G * np.sin(x[2]) * np.cos(del_) +
				 DoublePendulum.M2*DoublePendulum.L2 * x[3] * x[3] * np.sin(del_)
			   - (DoublePendulum.M1+DoublePendulum.M2)*DoublePendulum.G * np.sin(x[0]))/den1 + u - DoublePendulum.mu * x[1]

		dydx[2] = x[3]
		den2 = (DoublePendulum.L2/DoublePendulum.L1)*den1
		dydx[3] =  ((-DoublePendulum.M2*DoublePendulum.L2 * x[3]*x[3]*np.sin(del_)*np.cos(del_)
				   + (DoublePendulum.M1+DoublePendulum.M2)*DoublePendulum.G * np.sin(x[0])*np.cos(del_)
				   - (DoublePendulum.M1+DoublePendulum.M2)*DoublePendulum.L1 * x[1]*x[1]*np.sin(del_)
				   - (DoublePendulum.M1+DoublePendulum.M2)*DoublePendulum.G  * np.sin(x[2]))/den2 ) - DoublePendulum.mu * x[3]

		x[0] = x[0] + DoublePendulum.dt * dydx[0]
		x[1] = x[1] + DoublePendulum.dt * dydx[1]
		x[2] = x[2] + DoublePendulum.dt * dydx[2]
		x[3] = x[3] + DoublePendulum.dt * dydx[3]


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
		f = h5py.File( fout_name,  'w' )
		f.create_dataset( 'x', data=self.records )
		f.create_dataset( 'x_len', data=self.record_counter )

class App:

	def __init__(self):
		pg.init()
		pg.font.init()

		self.size = (1280, 480)
		self.title = "Bot control"
		self.screen = None
		self.running = False
		self.marker = None
		self.filter = 0
		font_name = pg.font.get_default_font()
		self.font = pg.font.SysFont(font_name, 40)
		self.clock = pg.time.Clock()


	def run(self, bottype, fout_name ):

		#Display init
		self.screen = pg.display.set_mode(self.size, pg.DOUBLEBUF | pg.HWSURFACE)
		pg.display.set_caption(self.title)
		pg.mouse.set_visible( True )
		pg.key.set_repeat(200, 50)
		self.snapshot = pg.Surface(self.size).convert()

		# initialise controller
		if bottype == 'doublependulum':
			controller = DoublePendulum( 1/-5., 5. )
		elif bottype == 'cartpole':
			controller = CartPole( 1/-5., 5. )
		else:
			print( 'unknown controller: ' + bottype )
			return -1

		width, height = self.screen.get_size()
		x0i, y0i = width / 2, height / 2	


		#Main loop
		self.running = True
		while self.running:

			#Blitting
			self.update()
			self.screen.fill((0,0,0))


			u = 0.
			#Events
			for event in pg.event.get():
				if event.type == pg.KEYDOWN:
					if event.key == pg.K_LEFT:
						label = self.font.render("KEY LEFT", 1, (255,255,0))
						self.screen.blit( label, (100, 100))
						u = -USE_FORCE

					elif event.key == pg.K_RIGHT:
						label = self.font.render("KEY RIGHT", 1, (255,255,0))
						self.screen.blit( label, (300, 100))
						u = USE_FORCE

					elif event.key == pg.K_r:
						controller.start_recording()

					if event.key == pg.K_ESCAPE:
						self.running = False

				elif event.type == pg.QUIT:
					self.running = False

			controller.step( u )
			controller.update( self.font, self.screen )


		controller.save_data( fout_name )


	#Blitting
	def update(self):

		#FPS limit
		self.clock.tick(60)

		#Show FPS
		fps_count = str(int(self.clock.get_fps()))
		fps = self.font.render( fps_count, True, pg.Color("white") )
		self.screen.blit(fps, (self.size[0] - fps.get_size()[0], 0))

		pg.display.flip()


if __name__ == "__main__":

	if len( sys.argv ) < 3:
		print( 'bot.py cartpole|doublependulum output_file_name' )
		sys.exit()

	app = App()
	app.run( sys.argv[1], sys.argv[2] )

